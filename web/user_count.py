import ray
import time
import logging
from typing import Dict

@ray.remote(memory=128 * 1024 * 1024)
class UserCountActor:
    def __init__(self):
        """初始化用户计数器"""
        self.users: Dict[str, float] = {}  # 用户ID -> 最后活跃时间
        self.timeout = 30  # 30秒超时
        self.max_retries = 3  # 最大重试次数
        
    def _cleanup_inactive_users(self):
        """清理不活跃的用户"""
        try:
            current_time = time.time()
            inactive_users = [
                user_id for user_id, last_active 
                in self.users.items() 
                if current_time - last_active > self.timeout
            ]
            for user_id in inactive_users:
                del self.users[user_id]
        except Exception as e:
            logging.error(f"清理不活跃用户时出错: {str(e)}")
    
    def heartbeat(self, user_id: str) -> int:
        """更新用户活跃状态
        
        Args:
            user_id: 用户ID
            
        Returns:
            int: 当前活跃用户数
        """
        try:
            self.users[user_id] = time.time()
            self._cleanup_inactive_users()
            return len(self.users)
        except Exception as e:
            logging.error(f"处理用户 {user_id} 心跳时出错: {str(e)}")
            return 0
            
    def get_count(self) -> int:
        """获取当前活跃用户数
        
        Returns:
            int: 当前活跃用户数
        """
        try:
            self._cleanup_inactive_users()
            return len(self.users)
        except Exception as e:
            logging.error(f"获取用户数量时出错: {str(e)}")
            return 0
