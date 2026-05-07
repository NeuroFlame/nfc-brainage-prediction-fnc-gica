import json
import logging

from nvflare.apis.impl.controller import Controller, Task
from nvflare.apis.controller_spec import ClientTask
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.shareable import Shareable
from typing import Callable

from _utils.utils import get_parameters_file_path

TASK_NAME_GET_LOCAL_SVR_WEIGHTS = "GET_LOCAL_SVR_WEIGHTS"
TASK_NAME_ACCEPT_AGGREGATED_WEIGHTS = "ACCEPT_AGGREGATED_WEIGHTS"
AGGREGATOR_ID = "aggregator"


class BrainAgeFNCController(Controller):
    def __init__(
        self,
        min_clients: int = 2,
        wait_time_after_min_received: int = 10,
        task_timeout: int = 0,
    ):
        """
        Initializes the BrainAgeFNCController.

        :param min_clients: Minimum number of client responses required per round.
        :param wait_time_after_min_received: Seconds to wait after min responses received.
        :param task_timeout: Per-task timeout in seconds (0 = no timeout).
        """
        super().__init__()
        self._task_timeout = task_timeout
        self._min_clients = min_clients
        self._wait_time_after_min_received = wait_time_after_min_received

#### Computation Author Defined Section ####
### This is where computation authors will define the control flow logic ###

    def start_controller(self, fl_ctx: FLContext) -> None:
        """
        Called when the controller starts. Assigns the aggregator component
        and loads computation parameters into the shared context.

        This is a Framework-Specific Required Method.

        :param fl_ctx: Federated learning context for this run.
        """
        self.aggregator = self._engine.get_component(AGGREGATOR_ID)
        self._load_and_set_computation_parameters(fl_ctx)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        """
        Defines the two-round federated SVR control flow.

        Round 0: Collect local SVR weight vectors from non-owner sites.
        Round 1: Send aggregated weights to all sites; owner trains projected SVR.

        This is the primary method that computation authors should focus on.

        :param abort_signal: Signal for aborting the flow if needed.
        :param fl_ctx: Federated learning context for this run.
        """
        # Round 0: broadcast local SVR training task; non-owner sites return weights
        self._broadcast_task(
            task_name=TASK_NAME_GET_LOCAL_SVR_WEIGHTS,
            data=Shareable(),
            result_cb=self._accept_local_svr_result,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        # Aggregate local weights into w_locals matrix
        aggregate_result = self.aggregator.aggregate(fl_ctx)

        # Round 1: broadcast aggregated weights; owner site trains projected SVR
        self._broadcast_task(
            task_name=TASK_NAME_ACCEPT_AGGREGATED_WEIGHTS,
            data=aggregate_result,
            result_cb=self._accept_owner_svr_result,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

    def _accept_local_svr_result(self, client_task: ClientTask, fl_ctx: FLContext) -> bool:
        """
        Callback: processes each site's round-0 result and sends it to the aggregator.

        :param client_task: The task result received from a client site.
        :param fl_ctx: Federated learning context for this run.
        :return: True if the result was successfully accepted.
        """
        return self.aggregator.accept(client_task.result, fl_ctx)

    def _accept_owner_svr_result(self, client_task: ClientTask, fl_ctx: FLContext) -> bool:
        """
        Callback: processes the owner site's round-1 result and sends it to the aggregator.
        Non-owner sites return empty Shareables and are silently ignored.

        :param client_task: The task result received from a client site.
        :param fl_ctx: Federated learning context for this run.
        :return: True if the result was successfully accepted.
        """
        return self.aggregator.accept(client_task.result, fl_ctx)

#### End of Computation Author Defined Section ####

#### Framework Helper Methods: No modification necessary ####

    def _broadcast_task(
        self,
        task_name: str,
        data: Shareable,
        result_cb: Callable[[ClientTask, FLContext], bool],
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> None:
        """
        Broadcasts a task to all client sites and waits for responses.

        :param task_name: Name of the task to broadcast.
        :param data: Shareable object containing the data to send.
        :param result_cb: Callback for handling results from each client site.
        :param fl_ctx: Federated learning context for this run.
        :param abort_signal: Signal used to abort the task if needed.
        """
        self.broadcast_and_wait(
            task=Task(
                name=task_name,
                data=data,
                props={},
                timeout=self._task_timeout,
                result_received_cb=result_cb,
            ),
            min_responses=self._min_clients,
            wait_time_after_min_received=self._wait_time_after_min_received,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

    def _load_and_set_computation_parameters(self, fl_ctx: FLContext) -> None:
        """
        Loads computation parameters from a file and sets them in the shared context
        for all sites to access.

        :param fl_ctx: Federated learning context for this run.
        """
        with open(get_parameters_file_path(fl_ctx), "r") as f:
            fl_ctx.set_prop(
                key="COMPUTATION_PARAMETERS",
                value=json.load(f),
                private=False,
                sticky=True,
            )

#### Framework-Specific Required Methods: No modification necessary ####

    def process_result_of_unknown_task(self, task: Task, fl_ctx: FLContext) -> None:
        """
        Handles results for tasks that are not explicitly recognized.

        This is a Framework-Specific Required Method.

        :param task: The task whose result is being processed.
        :param fl_ctx: Federated learning context for this run.
        """
        pass

    def stop_controller(self, fl_ctx: FLContext) -> None:
        """
        Called when the controller stops.

        This is a Framework-Specific Required Method.

        :param fl_ctx: Federated learning context for this run.
        """
        pass
