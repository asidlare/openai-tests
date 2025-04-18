from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, crew, task, tool
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
from llm.utils import get_openai_api_key, get_openai_model, get_serper_api_key
from llm.crewai_tests.tasks_event_planning.src.tasks_event_planning.types import VenueDetails

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class TasksEventPlanning():
    """TasksEventPlanning crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    llm = ChatOpenAI(
        openai_api_key=get_openai_api_key(),
        model_name=get_openai_model(),
    )

    @tool
    def serper_dev_tool(self) -> SerperDevTool:
        return SerperDevTool(
            api_key=get_serper_api_key(),
        )

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def venue_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config['venue_coordinator'],
            tools=[self.serper_dev_tool(), ScrapeWebsiteTool()],
            verbose=True
        )

    @agent
    def logistics_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['logistics_manager'],
            tools=[self.serper_dev_tool(), ScrapeWebsiteTool()],
            verbose=True
        )

    @agent
    def marketing_communications_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['marketing_communications_agent'],
            tools=[self.serper_dev_tool(), ScrapeWebsiteTool()],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def venue_task(self) -> Task:
        return Task(
            config=self.tasks_config['venue_task'],
            human_input=True,  # asks human for details
            output_json=VenueDetails,
            output_file="venue_details.json",
        )

    @task
    def logistics_task(self) -> Task:
        return Task(
            config=self.tasks_config['logistics_task'],
            human_input=True,  # asks human for details
            async_execution=True,  # ask can run in parallel with the tasks which come after it
        )

    @task
    def marketing_task(self) -> Task:
        return Task(
            config=self.agents_config['marketing_task'],
            async_execution=True,  # ask can run in parallel with the tasks which come after it
            output_file="marketing_report.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the TasksEventPlanning crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            verbose=True,
        )
