import json
import asyncio

from langchain_core.messages import HumanMessage

from app.store import data_store
from app.optimizer.schemas import Zone, Hospital
from app.optimizer.allocation import (
    build_graph, allocate_resources, events_to_context, RESOURCE_TYPES,
)
from app.config import load_hospitals_with_resources
from app.agents.llm import get_llm, is_ollama_available


CRISIS_ANALYSIS_PROMPT = (
    "You are a disaster crisis resolution analyst. Analyze the following "
    "crisis lifecycle data and provide:\n\n"
    "1. **Bottleneck Analysis**: Which zones were hardest to serve and why? "
    "Reference zone IDs, round numbers, and remaining demand.\n\n"
    "2. **Resource Efficiency**: Which resource types contributed the most "
    "demand coverage? Which hospitals were the primary contributors?\n\n"
    "3. **Event Impact**: How did events (road_collapse, weather) affect the "
    "resolution? Which routes were blocked and what alternatives were used?\n\n"
    "4. **Overall Narrative**: A concise 3-5 sentence narrative of how the "
    "crisis was resolved, including key turning points.\n\n"
    "Rules:\n"
    "- Be specific: reference zone IDs, hospital IDs, and resource types.\n"
    "- Keep each section to 2-3 sentences.\n"
    "- Write in English."
)


class CrisisManagerAgent:
    """Manages the crisis lifecycle using data_store directly (no HTTP).

    Usage:
        agent = CrisisManagerAgent()
        result = agent.run_lifecycle()  # returns full resolution summary
    """

    def __init__(self, max_rounds: int = 20):
        self.max_rounds = max_rounds

    def run_lifecycle(self) -> dict:
        """Run the full return → reoptimize loop until crisis is resolved.

        Returns rich structured data: per-round details, zone timelines,
        resource utilization, event impact, and optional LLM analysis.
        """
        crisis = data_store.get_crisis()
        if not crisis:
            return {"error": "No active crisis"}

        rounds = []
        zone_timelines = {}
        round_num = 0

        # Initialize zone timelines
        for zid, zs in crisis["zone_states"].items():
            zone_timelines[zid] = [{
                "round": 0,
                "event": "initial",
                "demand_before": zs["initial_demand"],
                "demand_after": zs["initial_demand"],
                "served_this_round": 0,
            }]

        while round_num < self.max_rounds:
            if data_store.is_crisis_resolved():
                break

            active = [
                d for d in crisis["dispatches"]
                if d["status"] == "dispatched"
            ]
            if not active:
                break

            # --- Per-zone demand snapshot BEFORE returns ---
            demand_before_by_zone = {
                zid: zs["remaining_demand"]
                for zid, zs in crisis["zone_states"].items()
            }
            demand_before_total = sum(demand_before_by_zone.values())

            # --- Batch return with detailed tracking ---
            returned_dispatches_detail = []
            for d in active:
                try:
                    capacity = RESOURCE_TYPES.get(
                        d["resource_type"], {},
                    ).get("capacity_per_unit", 1)
                    served = d["count"] * capacity
                    data_store.return_dispatch(d["dispatch_id"], served)
                    returned_dispatches_detail.append({
                        "dispatch_id": d["dispatch_id"],
                        "hospital_id": d["hospital_id"],
                        "zone_id": d["zone_id"],
                        "resource_type": d["resource_type"],
                        "count": d["count"],
                        "capacity_served": served,
                    })
                except ValueError:
                    pass

            # --- Per-zone demand snapshot AFTER returns ---
            demand_after_by_zone = {
                zid: zs["remaining_demand"]
                for zid, zs in crisis["zone_states"].items()
            }
            total_remaining = sum(demand_after_by_zone.values())

            # --- Per-zone demand deltas ---
            zone_demand_deltas = {}
            for zid in demand_before_by_zone:
                before = demand_before_by_zone[zid]
                after = demand_after_by_zone.get(zid, 0)
                zone_demand_deltas[zid] = {
                    "demand_before": before,
                    "demand_after": after,
                    "served_this_round": before - after,
                }

            bottleneck_zones = [
                zid for zid, delta in zone_demand_deltas.items()
                if delta["demand_after"] > 0
            ]

            # --- Update zone timelines ---
            for zid, delta in zone_demand_deltas.items():
                zone_timelines[zid].append({
                    "round": crisis["optimization_round"],
                    "event": "return",
                    "demand_before": delta["demand_before"],
                    "demand_after": delta["demand_after"],
                    "served_this_round": delta["served_this_round"],
                })

            # --- Build round info ---
            round_info = {
                "round": crisis["optimization_round"],
                "returned_dispatches": len(returned_dispatches_detail),
                "total_demand_served": demand_before_total - total_remaining,
                "remaining_demand_after_return": total_remaining,
                "dispatches_returned_detail": returned_dispatches_detail,
                "zone_demand_deltas": zone_demand_deltas,
                "bottleneck_zones": bottleneck_zones,
            }

            # Check if resolved after returns
            if data_store.is_crisis_resolved():
                round_info["action"] = "resolved_after_return"
                round_info["dispatches_created_detail"] = []
                rounds.append(round_info)
                break

            # --- Reoptimize ---
            unserved = data_store.get_unserved_zones()
            if not unserved:
                round_info["dispatches_created_detail"] = []
                rounds.append(round_info)
                break

            zones = [Zone(**z) for z in unserved]
            hospitals = [Hospital(**h) for h in load_hospitals_with_resources()]
            disaster_type = crisis.get("disaster_type")
            context = (
                events_to_context(crisis.get("events", []))
                if crisis.get("events") else {}
            )
            G = build_graph(zones, hospitals, context, disaster_type=disaster_type)
            new_results = allocate_resources(
                G, zones, hospitals, disaster_type=disaster_type,
            )

            dispatches_before_count = len(crisis["dispatches"])
            data_store.record_reoptimization(new_results)
            new_dispatches = crisis["dispatches"][dispatches_before_count:]

            round_info["action"] = "reoptimized"
            round_info["new_allocations"] = len(new_results)
            round_info["dispatches_created_detail"] = [
                {
                    "dispatch_id": d["dispatch_id"],
                    "hospital_id": d["hospital_id"],
                    "zone_id": d["zone_id"],
                    "resource_type": d["resource_type"],
                    "count": d["count"],
                    "capacity_served": d.get("capacity_served", d["count"]),
                    "status": d["status"],
                }
                for d in new_dispatches
            ]
            rounds.append(round_info)
            round_num += 1

        # --- Build final summary ---
        is_resolved = data_store.is_crisis_resolved()

        zone_summary = {}
        for zid, zs in crisis["zone_states"].items():
            zone_summary[zid] = {
                "initial_demand": zs["initial_demand"],
                "served": zs["served"],
                "remaining": zs["remaining_demand"],
            }

        # Finalize zone timelines
        for zid, timeline in zone_timelines.items():
            fully_served_round = None
            for entry in timeline:
                if entry["demand_after"] <= 0 and entry["event"] != "initial":
                    fully_served_round = entry["round"]
                    break
            zone_timelines[zid] = {
                "progression": timeline,
                "fully_served_at_round": fully_served_round,
            }

        resource_utilization = self._build_resource_utilization(crisis)
        event_impact_detail = self._build_event_impact(crisis)

        result_dict = {
            "resolved": is_resolved,
            "total_rounds": len(rounds),
            "rounds": rounds,
            "zone_summary": zone_summary,
            "zone_timelines": zone_timelines,
            "resource_utilization": resource_utilization,
            "event_impact_detail": event_impact_detail,
        }

        # LLM analysis (single call, optional)
        result_dict["crisis_manager_reasoning"] = self._generate_llm_analysis(
            result_dict, crisis.get("disaster_type", "unknown"),
        )

        if is_resolved:
            data_store.end_crisis()

        return result_dict

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_resource_utilization(self, crisis: dict) -> dict:
        """Build resource utilization summary from all dispatches."""
        hospital_contributions: dict[str, dict] = {}
        resource_type_totals: dict[str, dict] = {}

        for d in crisis["dispatches"]:
            if d["status"] in ("dispatched", "returned"):
                hid = d["hospital_id"]
                rtype = d["resource_type"]
                count = d["count"]
                cap = RESOURCE_TYPES.get(rtype, {}).get("capacity_per_unit", 1)

                hospital_contributions.setdefault(hid, {})
                hospital_contributions[hid][rtype] = (
                    hospital_contributions[hid].get(rtype, 0) + count
                )

                resource_type_totals.setdefault(
                    rtype, {"units_dispatched": 0, "total_capacity_served": 0},
                )
                resource_type_totals[rtype]["units_dispatched"] += count
                resource_type_totals[rtype]["total_capacity_served"] += count * cap

        sorted_resources = sorted(
            resource_type_totals.items(),
            key=lambda x: x[1]["total_capacity_served"],
            reverse=True,
        )
        most_effective = sorted_resources[0][0] if sorted_resources else None
        least_effective = sorted_resources[-1][0] if sorted_resources else None

        return {
            "hospital_contributions": hospital_contributions,
            "resource_type_totals": resource_type_totals,
            "most_effective_resource": most_effective,
            "least_effective_resource": least_effective,
        }

    def _build_event_impact(self, crisis: dict) -> dict:
        """Build event impact detail from crisis events."""
        events = crisis.get("events", [])
        if not events:
            return {"events_present": False}

        context = events_to_context(events)
        blocked_pairs = context.get("blocked_roads", [])
        weather_penalty = context.get("weather_penalty", 1.0)

        return {
            "events_present": True,
            "blocked_hospital_zone_pairs": blocked_pairs,
            "weather_penalty_value": weather_penalty,
            "road_block_penalty_value": context.get("road_block_penalty", 10.0),
            "summary": {
                "total_blocked_routes": len(blocked_pairs),
                "weather_affected": weather_penalty > 1.0,
                "blocked_hospitals": list({b["hospital_id"] for b in blocked_pairs}),
                "blocked_zones": list({b["zone_id"] for b in blocked_pairs}),
            },
        }

    def _generate_llm_analysis(self, lifecycle_data: dict, disaster_type: str) -> str:
        """Call LLM once to analyze the complete lifecycle."""
        try:
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    available = loop.run_in_executor(
                        pool, lambda: asyncio.run(is_ollama_available()),
                    )
            except RuntimeError:
                available = asyncio.run(is_ollama_available())

            if not available:
                return "LLM analysis unavailable: Ollama server not reachable."

            summary = {
                "disaster_type": disaster_type,
                "resolved": lifecycle_data["resolved"],
                "total_rounds": lifecycle_data["total_rounds"],
                "zone_summary": lifecycle_data["zone_summary"],
                "zone_timelines": {
                    zid: {
                        "fully_served_at_round": zt["fully_served_at_round"],
                        "total_steps": len(zt["progression"]),
                    }
                    for zid, zt in lifecycle_data.get("zone_timelines", {}).items()
                },
                "resource_utilization": lifecycle_data.get("resource_utilization", {}),
                "event_impact_detail": lifecycle_data.get("event_impact_detail", {}),
                "rounds_summary": [
                    {
                        "round": r["round"],
                        "total_demand_served": r["total_demand_served"],
                        "remaining": r["remaining_demand_after_return"],
                        "bottleneck_zones": r.get("bottleneck_zones", []),
                        "action": r.get("action", ""),
                    }
                    for r in lifecycle_data.get("rounds", [])
                ],
            }

            prompt = (
                f"Crisis lifecycle data:\n"
                f"{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
                f"Please analyze this crisis resolution."
            )

            llm = get_llm()
            response = llm.invoke([
                HumanMessage(content=f"{CRISIS_ANALYSIS_PROMPT}\n\n{prompt}"),
            ])
            content = response.content
            return content if isinstance(content, str) else str(content)

        except Exception as e:
            return f"LLM analysis unavailable: {e}"
