from typing import Dict, Any

from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.apis.fl_constant import ReservedKey

from .aggregate_weights import aggregate_local_svr_weights


class BrainAgeFNCAggregator(Aggregator):
    """
    Collects local SVR weight vectors from non-owner sites and aggregates
    them into the w_locals matrix sent to the owner site in round 1.

    Owner site results (is_owner=True) are stored separately and passed
    through to the controller for final output assembly.
    """

    def __init__(self):
        super().__init__()
        self.site_results: Dict[str, Dict[str, Any]] = {}
        self.owner_result: Dict[str, Any] = {}

    def accept(self, site_result: Shareable, fl_ctx: FLContext) -> bool:
        """
        Accept a result from a client site and store it for aggregation.

        :param site_result: Shareable containing either 'local_svr_result' (non-owner)
                            or 'owner_svr_result' (owner, round 1).
        :param fl_ctx: Federated learning context for this run.
        :return: True if the result was successfully accepted.
        """
        site_id = site_result.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default=None)
        computation_parameters = fl_ctx.get_prop(key="COMPUTATION_PARAMETERS", default={})
        site_id_name_map = computation_parameters.get("site_id_name_map", {})
        site_name = site_id_name_map.get(site_id, site_id)

        if site_result.get("is_owner"):
            # Owner sent a phase marker in round 0 — nothing to aggregate
            return True

        if "local_svr_result" in site_result:
            self.site_results[site_name] = site_result["local_svr_result"]
            return True

        if "owner_svr_result" in site_result:
            self.owner_result = site_result["owner_svr_result"]
            return True

        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """
        Aggregate local SVR weights from all non-owner sites into w_locals.

        :param fl_ctx: Federated learning context for this run.
        :return: Shareable containing 'w_locals' for broadcast to the owner site.
        """
        w_locals = aggregate_local_svr_weights(self.site_results)

        outgoing_shareable = Shareable()
        outgoing_shareable["w_locals"] = w_locals
        outgoing_shareable["local_site_results"] = self.site_results
        return outgoing_shareable
