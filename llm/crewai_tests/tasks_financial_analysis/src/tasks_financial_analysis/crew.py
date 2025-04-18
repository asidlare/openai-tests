from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, tool
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
from llm.utils import get_openai_api_key, get_openai_model, get_serper_api_key

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class TasksFinancialAnalysis():
    """TasksFinancialAnalysis crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    llm = ChatOpenAI(
        openai_api_key=get_openai_api_key(),
        model_name=get_openai_model(),
        temperature=0.7
    )

    @tool
    def serper_dev_tool(self) -> SerperDevTool:
        return SerperDevTool(
            api_key=get_serper_api_key(),
        )

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def data_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst_agent'],
            allow_delegation=True,
            tools=[ScrapeWebsiteTool(), self.serper_dev_tool()],
            verbose=True
        )

    @agent
    def trading_strategy_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['trading_strategy_agent'],
            allow_delegation=True,
            tools=[ScrapeWebsiteTool(), self.serper_dev_tool()],
            verbose=True
        )

    @agent
    def execution_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['execution_agent'],
            allow_delegation=True,
            tools=[ScrapeWebsiteTool(), self.serper_dev_tool()],
            verbose=True
        )

    @agent
    def risk_management_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['risk_management_agent'],
            allow_delegation=True,
            tools=[ScrapeWebsiteTool(), self.serper_dev_tool()],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def data_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_analysis_task'],
        )

    @task
    def strategy_development_task(self) -> Task:
        return Task(
            config=self.tasks_config['strategy_development_task'],
        )

    @task
    def execution_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['execution_planning_task'],
        )

    @task
    def risk_assessment_task(self) -> Task:
        return Task(
            config=self.tasks_config['risk_assessment_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the TasksFinancialAnalysis crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            manager_llm=self.llm,
            process=Process.hierarchical,  # helps to delegate the workflow to the Agents (kind of like a Manager at work)
            verbose=True,
        )
