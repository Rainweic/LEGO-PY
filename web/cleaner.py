import ray
import os
import shutil
import schedule
import time
from datetime import datetime


@ray.remote(num_cpus=0.1)
class Cleaner:
    def __init__(self, cache_dir="cache", max_days=7):
        # 缓存目录路径
        self.cache_dir = cache_dir
        # 文件最大保留天数
        self.max_days = max_days
    
    def clean(self):
        """清理超过指定天数未访问的缓存文件夹"""
        if not os.path.exists(self.cache_dir):
            return
            
        current_time = datetime.now()
        
        # 遍历缓存目录
        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            if not os.path.isdir(item_path):
                continue
                
            # 获取最后访问时间
            last_access_time = datetime.fromtimestamp(os.path.getatime(item_path))
            days_old = (current_time - last_access_time).days
            
            # 如果超过指定天数则删除
            if days_old > self.max_days:
                try:
                    shutil.rmtree(item_path)
                    print(f"已删除旧缓存目录: {item_path}")
                except Exception as e:
                    print(f"删除目录 {item_path} 时出错: {str(e)}")

    def run(self):
        # 立即执行一次清理
        self.clean()
        
        # 设置每天定时执行
        schedule.every().day.at("13:34").do(self.clean)

        while True:
            try:
                schedule.run_pending()
                # 移除调试用的print语句
                time.sleep(60 * 60 * 12)  # 每12小时检查一次
            except Exception as e:
                print(f"执行计划任务时出错: {str(e)}")
                time.sleep(60 * 60 * 12)  # 发生错误时等待12小时后继续


if __name__ == "__main__":
    try:
        cleaner = Cleaner.remote()
        # 使用 ray.get() 来等待远程任务完成
        ray.get(cleaner.run.remote())
    except KeyboardInterrupt:
        print("程序被用户终止")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")