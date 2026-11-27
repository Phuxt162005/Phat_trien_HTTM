from project01.agents.contracts import BaseAgent


def test_agent_message_contract():
    agent = BaseAgent("A")
    msg = agent.send("B", "PING", {"ok": True})
    assert msg.sender == "A"
    assert msg.receiver == "B"
    assert msg.message_type == "PING"
    assert msg.payload == {"ok": True}
    assert msg.message_id
