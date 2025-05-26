from pydantic import BaseModel, Field
from typing import Literal


class SalaryResponse(BaseModel):
    original_question: str = Field(
        description="""
            Original question asked by user in text form.
            Eg. 'How much money software developer at mid-level can earn today?'
        """
    )

    norm_title: Literal[
        'ml/ds',
        'data engineer',
        'manager',
        'data analyst',
        'bi',
        'research',
        'software developer',
        'ai engineer',
        'unknown'
    ] = Field(
        description=(
            """
                Normalized job title extracted from user's input question.
                Addresses the job title in a standarized form.
                Possible values:
                'ml/ds' (machine learning/data science),
                'data engineer' (data engineering),
                'manager' (management),
                'data analyst' (data analysis),
                'bi' (business intelligence),
                'research' (science research),
                'software developer' (software engineering, software development),
                'ai engineer' (artificial intelligence engineer),
                'unknown' (unknown job title / area)
            """
        )
    )

    level: Literal[
        'Entry-level',
        'Junior',
        'Intermediate',
        'Mid-level',
        'Senior-level',
        'Expert',
        'Director',
        'Executive-level',
        'unknown'
    ] = Field(
        ...,
        description=(
            """
                Experience level required or suggested for the question.
                Possible values:
                'Entry-level' (newbie),
                'Junior' (younger specialist),
                'Intermediate' (medium level specialist),
                'Mid-level' (medium level specialist),
                'Senior-level' (senior level specialist),
                'Expert' (expert level specialist),
                'Director' (director level, management level),
                'Executive-level' (board level, CEO level, higher management level),
                'unknown' (unknown experience level)
            """
        )
    )
    predicted_salary: float | None = Field(
        default=None,
        description="""
            Predicted salary based on a given question based on norm_title and experience level.
        """
    )
    extra_info: str | None = Field(
        default=None,
        description="""
            Extra information related to the predicted salary.
            Can be None if no extra information is available.
            If salary prediction is not possible, this field should contain a message indicating that. 
        """
    )


if __name__ == "__main__":
    response = SalaryResponse(
        original_question="How much money softwere developer at mid-level can earn today",
        norm_title="software",
        level="Mid-level"
    )
    print(response.model_dump())
