import asyncio
import json
import shutil
from pathlib import Path

from app.agent import HRAgent, WorkflowRequest
from hr_mcp.client import HRMCPClient

PROJECT_ROOT = Path(__file__).parents[1]


def test_mcp_tools_are_discoverable_and_callable() -> None:
    async def scenario() -> None:
        async with HRMCPClient() as client:
            tools = set(await client.list_tools())
            assert tools == {
                "search_policy_documents",
                "get_policy_section",
                "lookup_employee_profile",
                "check_pto_balance",
                "lookup_benefits_status",
                "check_policy_compliance",
                "create_mock_hr_ticket",
            }
            profile = await client.call_tool(
                "lookup_employee_profile", {"employee_id": "SYN-1001"}
            )
            assert profile["employee"]["name"] == "Avery Rowan"
            assert "email" not in profile["employee"]

    asyncio.run(scenario())


def test_remote_work_and_pto_workflows_use_mcp_tools() -> None:
    async def scenario() -> None:
        async with HRMCPClient() as client:
            agent = HRAgent(client)
            remote = await agent.run(
                WorkflowRequest(
                    workflow="remote_work",
                    employee_id="SYN-1001",
                    requested_location_id="US-NY",
                )
            )
            assert remote.status == "completed"
            assert remote.citations
            assert "[P1]" in remote.answer
            assert [step.tool for step in remote.trace if step.tool] == [
                "lookup_employee_profile",
                "search_policy_documents",
                "check_policy_compliance",
            ]

            pto = await agent.run(
                WorkflowRequest(
                    workflow="pto",
                    employee_id="SYN-1002",
                    requested_hours=8,
                )
            )
            assert pto.status == "completed"
            assert [step.tool for step in pto.trace if step.tool] == [
                "lookup_employee_profile",
                "check_pto_balance",
                "search_policy_documents",
                "check_policy_compliance",
            ]

    asyncio.run(scenario())


def test_workflow_clarifies_and_gates_mock_action_confirmation() -> None:
    async def scenario() -> None:
        async with HRMCPClient() as client:
            agent = HRAgent(client)
            clarification = await agent.run(
                WorkflowRequest(workflow="remote_work", employee_id="SYN-1001")
            )
            assert clarification.status == "needs_clarification"
            assert all(step.tool is None for step in clarification.trace)

            gated = await agent.run(
                WorkflowRequest(
                    workflow="pto",
                    employee_id="SYN-1002",
                    requested_hours=8,
                    create_ticket=True,
                )
            )
            assert gated.status == "confirmation_required"
            assert gated.mock_action
            assert not gated.mock_action["created"]
            assert gated.trace[-1].status == "confirmation_required"

    asyncio.run(scenario())


def test_confirmed_action_only_writes_to_configured_synthetic_store(tmp_path: Path) -> None:
    mock_data = tmp_path / "mock_data"
    shutil.copytree(PROJECT_ROOT / "mock_data", mock_data)

    async def scenario() -> None:
        async with HRMCPClient(env={"HR_MOCK_DATA_DIR": str(mock_data)}) as client:
            result = await HRAgent(client).run(
                WorkflowRequest(
                    workflow="pto",
                    employee_id="SYN-1002",
                    requested_hours=8,
                    create_ticket=True,
                    confirmed=True,
                )
            )
            assert result.status == "completed"
            assert result.mock_action
            assert result.mock_action["created"]
            assert result.mock_action["ticket"]["mock"] is True

    asyncio.run(scenario())
    original = json.loads((PROJECT_ROOT / "mock_data" / "tickets.json").read_text())
    copied = json.loads((mock_data / "tickets.json").read_text())
    assert len(copied["tickets"]) == len(original["tickets"]) + 1
