from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, tool
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool
from langchain_openai import ChatOpenAI
from llm.crewai_tests.l4_tools_customer_outreach.src.l4_tools_customer_outreach.tools.sentiment_analysis_tool import (
    SentimentAnalysisTool
)
from llm.utils import get_openai_api_key, get_openai_model, get_serper_api_key

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class L4ToolsCustomerOutreach():
    """L4ToolsCustomerOutreach crew"""

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
    def sales_rep_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['sales_rep_agent'],
            verbose=True
        )

    @agent
    def lead_sales_rep_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['lead_sales_rep_agent'],
            verbose=True
        )

    @tool
    def serper_dev_tool(self) -> SerperDevTool:
        return SerperDevTool(
            api_key=get_serper_api_key(),
        )

    @tool
    def directory_read_tool(self) -> DirectoryReadTool:
        return DirectoryReadTool(
            directory="./instructions"
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def lead_profiling_task(self) -> Task:
        return Task(
            config=self.tasks_config['lead_profiling_task'],
            tools=(self.directory_read_tool(), FileReadTool(), self.serper_dev_tool())
        )

    @task
    def personalized_outreach_task(self) -> Task:
        return Task(
            config=self.tasks_config['personalized_outreach_task'],
            tools=(SentimentAnalysisTool(), self.serper_dev_tool()),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the L4ToolsCustomerOutreach crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
        )
