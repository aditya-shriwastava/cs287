import argparse
import os
import logger
import json


def play():
    import cv2
    from envs import GridWorldEnv
    env1 = GridWorldEnv(seed=0)
    t = 0
    s_t = env1.reset()
    print(
        "Usage:\n" +\
        "* Use arrow keys to move\n" +\
        "* Esc to exit\n"
    )
    while True:
        img = env1.render(mode='rgb_array')
        cv2.imshow("Render", img)
        print(f"s_{t}: {s_t}")

        key_mapping = {82: 0, 84: 1, 81: 2, 83: 3} # Arrow keys: Up, Down, Left, Right
        key = cv2.waitKey(0) & 0xFF
        if key in key_mapping:
            a_t = key_mapping[key]
            print(f"a_t: {a_t}")
        elif key == 27:  # Break the loop if the 'Esc' key is pressed
            break
        else:
            print(f"Invalid key (i.e. {key}). Please press 'w', 's', 'a', or 'd'.")
            continue

        s_tp1, reward, done, env_info = env1.step(a_t)
        print("----------")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"env_info: {env_info}")
        print("----------")
        t += 1
        s_t = s_tp1
        if done:
            print("Done")
            break


def main(args):
    render = args.render
    if not render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    from utils.utils import TabularPolicy, TabularValueFun
    from part1.tabular_value_iteration import ValueIteration
    from envs import GridWorldEnv
    envs = [GridWorldEnv(seed=0)]

    for env in envs:
        env_name = env.__name__
        exp_dir = os.getcwd() + '/data/part1/%s/policy_type%s_temperature%s/' % (env_name, args.policy_type, args.temperature)
        logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'])
        args_dict = vars(args)
        args_dict['env'] = env_name
        json.dump(vars(args), open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True)

        policy = TabularPolicy(env)
        value_fun = TabularValueFun(env)
        algo = ValueIteration(env,
                              value_fun,
                              policy,
                              policy_type=args.policy_type,
                              render=render,
                              temperature=args.temperature)
        algo.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_type", "-p", type=str, default='deterministic', choices=["deterministic", "max_ent"],
                        help="Whether to train a deterministic policy or a maximum entropy one")
    parser.add_argument("--render", "-r", action='store_true', help="Vizualize the policy and contours when training")
    parser.add_argument("--temperature", "-t", type=float, default=1.,
                        help="Temperature parameter for maximum entropy policies")
    args = parser.parse_args()
    main(args)
    # play()
