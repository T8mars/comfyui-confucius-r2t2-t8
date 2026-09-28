import asyncio
import unittest

from r2t2_core.worker import (ACTIVE_IDLE_SECONDS, AWAITING_FIRST_AUDIO_SECONDS,
                              TERMINAL_RETENTION_SECONDS, LiveState, Service)


class FakeSession:
    pass


class WorkerLifecycleTests(unittest.TestCase):
    def test_idle_session_interrupts_and_is_later_reaped(self):
        service = Service()
        state = LiveState(FakeSession(), "owner", "sid")
        service.sessions[state.sid] = state
        base = state.updated

        async def exercise():
            await service.reap_idle(base + AWAITING_FIRST_AUDIO_SECONDS - 1)
            self.assertEqual(state.status, "active")
            await service.reap_idle(base + AWAITING_FIRST_AUDIO_SECONDS + 1)
            self.assertEqual(state.status, "interrupted")
            self.assertEqual(state.revision, 1)
            await service.reap_idle(state.updated + TERMINAL_RETENTION_SECONDS + 1)
            self.assertNotIn(state.sid, service.sessions)

        asyncio.run(exercise())

    def test_audio_session_has_shorter_idle_limit(self):
        service = Service()
        state = LiveState(FakeSession(), "owner", "sid", samples=640)
        service.sessions[state.sid] = state
        asyncio.run(service.reap_idle(state.updated + ACTIVE_IDLE_SECONDS + 1))
        self.assertEqual(state.status, "interrupted")


if __name__ == "__main__":
    unittest.main()
