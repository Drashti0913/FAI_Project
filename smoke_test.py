"""
Quick smoke test — run this first to verify everything is wired correctly.
Should complete in under 10 seconds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.delivery_env import DeliveryEnv
from agents.baselines import RandomBaseline, RoundRobinBaseline, GreedyNearestBaseline
from agents.q_agent import QLearningAgent


def test_city():
    from environment.city import City
    city = City(traffic_pattern="rush_hour", seed=42)
    assert len(city.edges) == 40, f"Expected 40 edges, got {len(city.edges)}"
    path, cost = city.shortest_path(0, 24, hour=9)
    assert path[0] == 0 and path[-1] == 24, "Path should go from 0 to 24"
    assert cost > 0
    print(f"  [OK] City: 40 edges, path 0→24 cost={cost:.1f}, len={len(path)}")


def test_order_gen():
    from environment.order import OrderGenerator
    gen = OrderGenerator(arrival_rate=1.0, seed=42)  # always generates
    order = gen.step(10)
    assert order is not None
    assert order.deadline == 10 + 30
    print(f"  [OK] Order: {order}")


def test_driver():
    from environment.city import City
    from environment.driver import Driver
    from environment.order import Order

    city = City(seed=42)
    driver = Driver(id=0, current_node=0)
    order = Order(id=0, destination=6, arrival_time=0, deadline=30)
    orders = {0: order}

    driver.assign_order(0, 0)
    driver.recompute_route(orders, city, hour=9)
    assert len(driver.route) > 0, "Route should be non-empty"
    print(f"  [OK] Driver: route to node 6 = {[0] + driver.route}")


def test_env_episode():
    env = DeliveryEnv(traffic_pattern="rush_hour", seed=42)
    baseline = RandomBaseline()
    state = env.reset()

    steps = 0
    while not env.done and steps < 2000:
        if env.pending_orders:
            action_idx = baseline.act(state, env)
            state, reward, done, info = env.step(env.decode_action(action_idx))
        else:
            env._run_until_decision()
        steps += 1

    assert env.done, "Episode should have terminated"
    print(f"  [OK] Episode: {env.completed_count} orders, reward={info.get('final_reward', '?')}")


def test_q_agent():
    env = DeliveryEnv(seed=42)
    agent = QLearningAgent()
    state = env.reset()
    state_key = env.encode_state()

    steps = 0
    while not env.done and steps < 200:
        if env.pending_orders:
            action = agent.act(state_key, env)
            next_state, reward, done, info = env.step(env.decode_action(action))
            next_key = env.encode_state()
            agent.update(state_key, action, reward, next_key, done, env.valid_actions())
            state_key = next_key
        else:
            break
        steps += 1

    agent.decay_epsilon()
    print(f"  [OK] Q-agent: epsilon={agent.epsilon:.3f}, states={len(agent.q_table)}")


if __name__ == "__main__":
    print("Running smoke tests...\n")
    tests = [test_city, test_order_gen, test_driver, test_env_episode, test_q_agent]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("All good — run: python training/train.py")