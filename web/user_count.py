import ray
from datetime import datetime

@ray.remote
class UserCountActor:
    def __init__(self):
        self.active_users = {}  # 使用字典存储用户ID和最后活跃时间
        self.timeout = 30  # 30秒超时

    def heartbeat(self, user_id):
        """更新用户的最后活跃时间"""
        self.active_users[user_id] = datetime.now()
        self._cleanup()
        return len(self.active_users)

    def _cleanup(self):
        """清理超时的用户"""
        now = datetime.now()
        self.active_users = {
            uid: last_active 
            for uid, last_active in self.active_users.items()
            if (now - last_active).total_seconds() < self.timeout
        }

    def get_count(self):
        """获取当前活跃用户数"""
        self._cleanup()
        return len(self.active_users)
