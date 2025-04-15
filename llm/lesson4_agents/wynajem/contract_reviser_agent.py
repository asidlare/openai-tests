from datetime import datetime
from llm.lesson4_agents.wynajem.base_agent import (
    BaseAgent,
    AuditResult,
)


class ContractReviserAgent(BaseAgent):
    def run(self) -> bool:
        self.logger.info(f"[ContractReviserAgent] Revising contract... Version: {self.context.metadata.current_version}")

        if not self.context.metadata.audit_history:
            self.logger.info("[ContractReviserAgent] No audit history found. Nothing to revise.")
            return True  # Jeśli brak historii audytów, zwróć sukces

        last_audit = self.context.metadata.audit_history[-1]

        if self._apply_changes(last_audit):
            self.context.metadata.current_version += 1
            self.context.metadata.current_revision_attempt += 1
            self.context.metadata.last_update_time = datetime.now()
            return True
        return False

    def _apply_changes(self, last_audit: AuditResult) -> bool:
        return True
