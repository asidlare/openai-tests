from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from llm.utils import get_openai_api_key, get_openai_model


@CrewBase
class BlogArticle():
    """BlogArticle crew"""

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
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config['planner'],
            allow_delegation=False,
            verbose=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'],
            allow_delegation=False,
            verbose=True
        )

    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config['editor'],
            allow_delegation=False,
            verbose=True,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def plan(self) -> Task:
        return Task(
            config=self.tasks_config['plan'],
        )

    @task
    def write(self) -> Task:
        return Task(
            config=self.tasks_config['write'],
        )

    @task
    def edit(self) -> Task:
        return Task(
            config=self.tasks_config['edit'],
            output_file='article.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the BlogArticle crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
