import asyncio
import json
import shutil
from pathlib import Path

from app.agent import ChatRequest, HRAgent, WorkflowRequest
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
                "list_work_locations",
            }
            profile = await client.call_tool(
                "lookup_employee_profile", {"employee_id": "SYN-1001"}
            )
            assert profile["employee"]["name"] == "Avery Rowan"
            assert "email" not in profile["employee"]
            missing = await client.call_tool(
                "lookup_employee_profile", {"employee_id": "SYN-9999"}
            )
            assert missing["found"] is False
            assert "SYN-9999" in missing["reason"]
            unknown_section = await client.call_tool(
                "get_policy_section",
                {"source": "remote-work.md", "section": "Not a real section"},
            )
            assert unknown_section["found"] is False

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


def test_workflow_uses_configured_llm_synthesizer() -> None:
    class FakeSynthesizer:
        async def synthesize(self, evidence: dict[str, object]) -> str:
            assert evidence["decision"] == "eligible_for_review"
            return "Model-generated grounded guidance. [P1]"

    async def scenario() -> None:
        async with HRMCPClient() as client:
            result = await HRAgent(client, synthesizer=FakeSynthesizer()).run(
                WorkflowRequest(
                    workflow="remote_work",
                    employee_id="SYN-1001",
                    requested_location_id="US-NY",
                )
            )
            assert result.answer == "Model-generated grounded guidance. [P1]"
            assert any(
                step.state == "synthesize" and "configured LLM" in step.result_summary
                for step in result.trace
            )

    asyncio.run(scenario())


def test_chat_follow_up_supplies_hours_and_confirms_ticket(tmp_path: Path) -> None:
    mock_data = tmp_path / "mock_data"
    shutil.copytree(PROJECT_ROOT / "mock_data", mock_data)

    async def scenario() -> None:
        async with HRMCPClient(env={"HR_MOCK_DATA_DIR": str(mock_data)}) as client:
            agent = HRAgent(client)
            clarification = await agent.chat(ChatRequest(message="Can SYN-1002 take PTO?"))
            assert clarification.status == "needs_clarification"
            assert clarification.context is not None

            completed = await agent.chat(
                ChatRequest(message="8 hours", context=clarification.context)
            )
            assert completed.status == "completed"
            assert completed.workflow == "pto"

            proposed = await agent.chat(
                ChatRequest(
                    message="SYN-1002 wants 4 hours of PTO and a mock HR ticket."
                )
            )
            assert proposed.status == "confirmation_required"
            assert proposed.context is not None
            assert proposed.context.awaiting_confirmation

            confirmed = await agent.chat(
                ChatRequest(message="Yes, create the ticket", context=proposed.context)
            )
            assert confirmed.status == "completed"
            assert confirmed.mock_action is not None
            assert confirmed.mock_action["created"] is True

    asyncio.run(scenario())


def test_unknown_place_is_reported_with_supported_locations() -> None:
    async def scenario() -> None:
        async with HRMCPClient() as client:
            agent = HRAgent(client)
            result = await agent.chat(
                ChatRequest(message="Can SYN-1001 work remotely from Oregon?")
            )
            assert result.status == "escalated"
            assert "Oregon" in result.answer
            assert "California" in result.answer
            assert "New York" in result.answer
            assert "Texas" in result.answer
            assert "Please provide the requested work location" not in result.answer
            assert "list_work_locations" in [step.tool for step in result.trace]

            listed = await agent.chat(
                ChatRequest(message="Where can I work remotely?")
            )
            assert listed.status == "completed"
            assert "California" in listed.answer
            assert "list_work_locations" in [step.tool for step in listed.trace]

            clarification = await agent.chat(
                ChatRequest(message="Can SYN-1001 work remotely?")
            )
            follow_up = await agent.chat(
                ChatRequest(message="Seattle", context=clarification.context)
            )
            assert follow_up.status == "escalated"
            assert "Seattle" in follow_up.answer
            assert "California" in follow_up.answer

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
