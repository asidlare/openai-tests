from llm.lesson4_agents.wynajem.base_agent import ContractStatus, ProcessContext, logger
from llm.lesson4_agents.wynajem.data_collector_agent import DataCollectorAgent
from llm.lesson4_agents.wynajem.contract_generator_agent import ContractGeneratorAgent
from llm.lesson4_agents.wynajem.contract_auditor_agent import ContractAuditorAgent
from llm.lesson4_agents.wynajem.contract_reviser_agent import ContractReviserAgent


'''
             +-----------------------+
             |   COLLECTING_DATA     |
             +-----------------------+
                        |
                        v  # 1
             +-----------------------+
             |      GENERATING       |
             +-----------------------+
                        |
                        v  # 2
             +-----------------------+
             |       AUDITING        |
             +-----------------------+
               |               |
               | Success  # 3   | Failure  # 5
               v               v
        +-----------------+    +-------------------+
        |   COMPLETED     |    |     REVISING      |
        +-----------------+    +-------------------+
                  ^                   |     ^
                  | Success   # 4     v     | Failure  # 6
                  |        +-----------------------+
                  ---------|       AUDITING        |
                           +-----------------------+


Jeśli wystąpi błąd lub nie znaleziono agenta:
                 +----------------+
                 |     ERROR      |  # 7
                 +----------------+
'''


class ContractCoordinator:
    def __init__(self):
        self.process_context = ProcessContext()

        self.agents = {
            ContractStatus.COLLECTING_DATA: DataCollectorAgent(context=self.process_context),
            ContractStatus.GENERATING: ContractGeneratorAgent(context=self.process_context),
            ContractStatus.AUDITING: ContractAuditorAgent(context=self.process_context),
            ContractStatus.REVISING: ContractReviserAgent(context=self.process_context),
        }
        self.logger = logger

    def process_contract(self) -> bool:
        while self.process_context.metadata.status not in (ContractStatus.COMPLETED, ContractStatus.ERROR):
            current_agent = self.agents.get(self.process_context.metadata.status)

            if current_agent is None:
                self.process_context.metadata.status = ContractStatus.ERROR
                self.logger.error("Unknown agent!")
                return False

            success = current_agent.run()

            if not success:
                if self.process_context.metadata.status == ContractStatus.AUDITING:
                    if self.process_context.metadata.current_revision_attempt >= self.process_context.metadata.max_revision_attempts:
                        self.logger.error("Przekroczono maksymalną liczbę prób rewizji")
                        self.process_context.metadata.status = ContractStatus.ERROR
                        return False

                    self.process_context.metadata.status = ContractStatus.REVISING
                else:
                    self.process_context.metadata.status = ContractStatus.ERROR
                    return False
            else:
                self._update_status()

    def _update_status(self):
        status_flow = {
            ContractStatus.COLLECTING_DATA: ContractStatus.GENERATING,  # 1
            ContractStatus.GENERATING: ContractStatus.AUDITING,  # 2
            ContractStatus.AUDITING: ContractStatus.COMPLETED,  # 3, 4
            ContractStatus.REVISING: ContractStatus.AUDITING,  # 6
        }

        self.process_context.metadata.status = status_flow.get(  # 7
            self.process_context.metadata.status,
            ContractStatus.ERROR,
        )


if __name__ == "__main__":
    agent = ContractCoordinator()
    agent.process_contract()
