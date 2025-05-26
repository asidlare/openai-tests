from mcp.server.fastmcp import FastMCP
# from mcp.server.fastmcp.prompts import base
from salary_model.salary import predict
# from mcp import tool


mcp = FastMCP("Salary Prediction Server")


@mcp.tool(
    name="Salary Prediction Tool",
    description="Predicts salary based on job norm title and level."
)
def predict_salary(norm_title: str, level: str) -> float:
    sample = {'norm_title': norm_title, 'level': level}
    return predict(sample) * 1000

# Example usage
if __name__ == "__main__":
    mcp.run()
    # predicted_salary = predict_salary('Software Developer', 'Mid-level')
    # print(f'Predicted salary: {predicted_salary * 1000} USD yearly')
