from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, tool
from crewai_tools import ScrapeWebsiteTool
from langchain_openai import ChatOpenAI
from llm.utils import get_openai_api_key, get_openai_model

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class L3CustomerSupport():
    """L3CustomerSupport crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    llm = ChatOpenAI(
        openai_api_key=get_openai_api_key(),
        model_name=get_openai_model(),
    )

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_agent'],
            allow_delegation=False,
            verbose=True
        )

    @agent
    def support_quality_assurance_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_quality_assurance_agent'],
            verbose=True
        )

    @tool
    def crews_scrape_tool(self) -> ScrapeWebsiteTool:
        return ScrapeWebsiteTool(
            website_url="https://docs.crewai.com/concepts/crews"
        )

    @tool
    def memory_scrape_tool(self) -> ScrapeWebsiteTool:
        return ScrapeWebsiteTool(
            website_url="https://docs.crewai.com/concepts/memory"
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def inquiry_resolution(self) -> Task:
        return Task(
            config=self.tasks_config['inquiry_resolution'],
            tools=[self.crews_scrape_tool(), self.memory_scrape_tool()]
        )

    @task
    def quality_assurance_review(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_review'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the L3CustomerSupport crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
        )
