# -*- coding:utf-8  -*-
# 作者：zruizhi
# 创建时间： 2020/9/11 11:17 上午
# 描述：选择运行环境，需要维护env/__ini__.py && config.json（存储环境默认参数）

import json
import env
import os


def make(env_type, seed=None, conf=None, game_name="olympic-integrated"):
    print(f"{env_type = }")
    file_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not conf:
        with open(file_path) as f:
            conf = json.load(f)[env_type]
    class_literal = conf['class_literal']
    if env_type.split('-')[0] in ["olympics"]:
        print(f"!!{game_name}")
        return getattr(env, class_literal)(conf, seed, game_name)
    else:
        return getattr(env, class_literal)(conf, game_name)


if __name__ == "__main__":
    make("olympics_running")