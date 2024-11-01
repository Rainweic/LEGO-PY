# -*- coding: utf-8 -*-
import subprocess
import os
import shutil
import multiprocessing
from datetime import datetime
import logging
from tqdm import tqdm  # 导入tqdm库


try:
    """
    由于物理机和idata平台需要执行的脚本不同
    环境信息加载请放在前面，否则会报pydoop 或 hdfs 加载异常
    """
    # 本地shell脚本
    cmd_local_file = "/opt/mrsclient/bigdata_env"
    source_cmd_local = "source " + cmd_local_file + " && env"

    # pyspark on yarn
    source_cmd_yarn = "source /opt/pydoop-install/mrsclient/bigdata_env && env"
    if os.path.isfile(cmd_local_file):
        print(source_cmd_local)
        result = subprocess.run(
            source_cmd_local, shell=True, check=True, stdout=subprocess.PIPE, executable='/bin/bash'
        )
        output = result.stdout.decode("utf-8")
    else:
        print(source_cmd_yarn)
        result = subprocess.run(
            source_cmd_yarn, shell=True, check=True, stdout=subprocess.PIPE, executable='/bin/bash'
        )
        output = result.stdout.decode("utf-8")


    for line in output.splitlines():
        var = line.strip().split("=")
        if len(var) == 2:
            os.environ[var[0]] = var[1]

    import pydoop.hdfs as hdfs

except BaseException as e:
    logging.error(e)
    pass


class InsecureClient:

    def __init__(self, _bucket_name=None, _user=None, _groups=None, **kwargs):
        
        self.user = _user
        self.fs = hdfs.hdfs(user=_user)
        if not _bucket_name.endswith("/"):
            _bucket_name += "/"

        self.bucket_name = _bucket_name
        self.logger = logging.getLogger("demo")
        self.logger.setLevel(logging.INFO)

    def is_file_local(self, local_path):
        self.logger.info("Determining local %s is a file or path...", local_path)
        if os.path.isfile(local_path):
            self.logger.info("Local %s is a file.", local_path)
            return True
        else:
            self.logger.info("Local %s is a folder.", local_path)
            return False

    def list_file_local(self, local_path):
        self.logger.info("Finding all objects under path %s locally.", local_path)
        result_list = []
        for root, dirs, files in os.walk(local_path):
            for file in files:
                result_list.append(os.path.join(root, file))
        self.logger.info("All files found locally are:")
        self.logger.info(result_list)
        return result_list

    def delete_local(self, local_path):
        self.logger.info("Start to delete local path %s .", local_path)
        os.system("rm -rf " + local_path)
        self.logger.info("Delete local path %s succeeded.", local_path)
        return True

    def status(self, hdfs_path, strict=True):
        """
        修改人：杨卓士
        修改时间：2024-06-13
        修改内容：新方法中status返回的内容与原方法不同，需要将新方法的内容变为字典，并将key按照原方法进行修改
        """
        try:
            status = self.fs.get_path_info(hdfs_path)
            status["modificationTime"] = status["last_mod"]
            status["length"] = status["size"]
            self.logger.info(f"Get OBS object status succeeded: {hdfs_path}")
            return status
        except IOError as e:
            self.logger.warning(f"Get object status failed, may not exist: {hdfs_path}")
            self.logger.warning(f"Error: {e}")
            return None

    def hdfs_path_stat(self, hdfs_path):
        return self.fs.get_path_info(hdfs_path)

    def hdfs_path_exists(self, hdfs_path):
        return self.fs.exists(hdfs_path)

    def hdfs_path_isfile(self, hdfs_path):
        stats = self.fs.get_path_info(hdfs_path)
        return stats["kind"] == "file"

    def hdfs_path_isdir(self, hdfs_path):
        stats = self.fs.get_path_info(hdfs_path)
        return stats["kind"] == "directory"

    def hdfs_rmr(self, hdfs_path):
        hdfs.rmr(hdfs_path, user=self.user)

    def hdfs_put(self, local_path: str, hdfs_path: str, overwrite: bool):
        self.upload_single_file(local_path, hdfs_path, overwrite)

    def delete_remote_path(self, hdfs_path, obs_root_path):
        for file_path in self.fs.list_directory(hdfs_path):
            file_path_str = os.path.join(
                "/", file_path["path"].replace(obs_root_path, "")
            )
            self.hdfs_rmr(file_path_str)

    def upload_single_file(self, local_path, hdfs_path, overwrite):
        print(
            "upload_single_file===================== start time is:"
            + format(datetime.now())
        )
        hdfs_path_full = hdfs_path + "/" + os.path.basename(local_path)
        if not overwrite and self.fs.exists(hdfs_path_full):
            self.logger.error(
                f"the OBS target path {hdfs_path_full} is already existed as a file.",
                hdfs_path_full,
            )
            return None

        hdfs_path_final = hdfs_path + "/" + os.path.basename(local_path)

        with hdfs.open(hdfs_path_final, "wb", user=self.user) as final_file:
            with open(local_path, "rb") as local_file:
                while True:
                    data = local_file.read(1 * 1024 * 1024 * 100)
                    if not data:
                        break
                    final_file.write(data)
        self.logger.info(
            "Upload local file %s to OBS %s succeeded.", local_path, hdfs_path
        )
        print(
            "upload_single_file===================== end time is:"
            + format(datetime.now())
        )
        return hdfs_path_final

    def download_single_file(self, hdfs_path, local_path, overwrite):
        try:
            if os.path.exists(local_path):
                download_file_name = hdfs_path.split("/")[-1]
                local_file = local_path + "/" + download_file_name
                if overwrite and os.path.exists(local_file):
                    os.remove(local_file)

                if not overwrite and os.path.exists(local_file):
                    raise ValueError(
                        "The target path " + local_file + " is already exists."
                    )

                with open(local_file, "wb") as final_file:
                    with hdfs.open(hdfs_path, "rb", user=self.user) as hdfs_file:
                        while True:
                            data = hdfs_file.read(1 * 1024 * 1024 * 256)
                            if not data:
                                break
                            final_file.write(data)
                return hdfs_path
            else:
                raise ValueError("The target path " + local_path + " not exists.")
        except Exception as e:
            self.logger.error(
                "Download OBS file failed: " + hdfs_path + "Error: " + str(e)
            )
            return None

    def upload_folder(self, local_path, hdfs_path, overwrite):
        # 如果没有，则添加"/"
        if not hdfs_path.endswith("/"):
            hdfs_path += "/"

        # 校验HDFS目录是否存在
        hdfs_path_file = hdfs_path + os.path.basename(local_path)
        if not self.hdfs_path_exists(hdfs_path_file):
            self.fs.create_directory(hdfs_path_file)

        # 遍历本地目录中的文件及子目录
        for root, dirs, files in os.walk(local_path):
            # 在HDFD上构造与本地目录相同的目录及子目录
            hdfs_root = hdfs_path_file + "/" + os.path.relpath(root, local_path)
            # 如果目录在HDFS中不存在，则创建它
            if not self.hdfs_path_exists(hdfs_root):
                self.fs.create_directory(hdfs_root)

            # 上传文件
            for file in files:
                local_file_path = os.path.join(root, file)
                hdfs_file_path = hdfs_root + "/" + file
                # 判断是否要覆盖已存在文件
                if overwrite and self.hdfs_path_exists(hdfs_file_path):
                    self.hdfs_rmr(hdfs_file_path)
                self.upload_single_file(local_file_path, hdfs_file_path, overwrite)
        self.logger.info("upload folder %s to path %s success", local_path, hdfs_path)

    def clear_folder(self, local_path, hdfs_path):
        # 遍历HDFS目录中的文件和子目录
        hdfs_path_full = hdfs_path + "/" + os.path.basename(local_path)
        for filename in self.fs.list_directory(hdfs_path_full):
            hdfs_list = os.path.join(
                hdfs_path_full, filename["name"].replace(self.bucket_name, "/")
            )
            self.hdfs_rmr(hdfs_list)

    def upload_folder_force(self, local_path, hdfs_path, overwrite):
        # 如果没有，则添加"/"
        if not hdfs_path.endswith("/"):
            hdfs_path += "/"
        if overwrite and self.hdfs_path_exists(hdfs_path):
            self.clear_folder(local_path, hdfs_path)
        self.upload_folder(local_path, hdfs_path, overwrite)
        self.logger.info(
            "force upload folder local_path %s to hdfs_path %s success.",
            local_path,
            hdfs_path,
        )
        return None

    def download_folder(self, hdfs_path, local_path, overwrite):
        # 确保本地目录存在
        local_path_file_exists = (
            os.path.normpath(local_path) + "/" + os.path.basename(hdfs_path)
        )
        if not os.path.exists(local_path_file_exists):
            os.makedirs(local_path_file_exists)

        # 遍历HDFS目录中的文件和子目录
        for filename in self.fs.list_directory(hdfs_path):
            hdfs_list = os.path.join(
                hdfs_path, filename["name"].replace(self.bucket_name, "/")
            )
            local_list = os.path.join(
                local_path_file_exists, os.path.basename(filename["name"])
            )
            # 若是文件，直接下载
            if self.hdfs_path_isfile(hdfs_list):
                # 判断文件是否已存在，是否需要覆盖
                if os.path.exists(local_list) and overwrite:
                    os.remove(local_list)
                resp = self.download_single_file(
                    hdfs_list, os.path.dirname(local_list), overwrite
                )
            # 若是文件夹，则递归下载
            elif self.hdfs_path_isdir(hdfs_list):
                self.download_folder(hdfs_list, local_path_file_exists, overwrite)

    def download_folder_force(self, hdfs_path, local_path, overwrite):
        # 如果没有，则添加"/"
        if not local_path.endswith("/"):
            local_path += "/"

        local_path_full = local_path + os.path.basename(hdfs_path)
        if overwrite and os.path.exists(local_path_full):
            shutil.rmtree(local_path_full)

        self.download_folder(hdfs_path, local_path, overwrite)
        self.logger.info(
            "force download folder hdfs_path %s to local_path %s success.",
            hdfs_path,
            local_path,
        )
        return None

    def upload_file(self, local_file, hdfs_path, overwrite):
        if not hdfs_path.endswith("/"):
            hdfs_path += "/"
        self.hdfs_put(local_file, hdfs_path, overwrite)
        return hdfs_path

    def create_folder(self, hdfs_path):
        if not self.hdfs_path_exists(hdfs_path):
            self.fs.create_directory(hdfs_path)


insecure_client = None
multiprocessing_mode = None


def set_start_multiprocess():
    global multiprocessing_mode

    # 如果启动方法还未设置或者需要设置为'spawn'，则进行设置
    if multiprocessing_mode != "spawn":
        try:
            multiprocessing.set_start_method("spawn")
            multiprocessing_mode = "spawn"
            print("Start method set to 'spawn'.")
        except RuntimeError as e:
            print(f"Failed to set start method: {e}")


def process_dirs(local_path: str, hdfs_path: str, obs_root_path: str, hdfs_user: str):
    global insecure_client
    if not insecure_client:
        insecure_client = InsecureClient(obs_root_path, _user=hdfs_user)
    insecure_client.create_folder(hdfs_path)


def process_files(
    local_path: str, hdfs_path: str, obs_root_path: str, hdfs_user: str, overwrite: bool
):
    global insecure_client
    if not insecure_client:
        insecure_client = InsecureClient(obs_root_path, _user=hdfs_user)

    insecure_client.upload_file(local_path, hdfs_path, overwrite)


def process_if_overwrite(
    hdfs_path: str, obs_root_path: str, overwrite: bool, hdfs_user: str
):
    global insecure_client
    if not insecure_client:
        insecure_client = InsecureClient(obs_root_path, _user=hdfs_user)

    if insecure_client.hdfs_path_exists(hdfs_path):
        if overwrite:
            insecure_client.delete_remote_path(hdfs_path, obs_root_path)
    else:
        insecure_client.create_folder(hdfs_path)


def multi_process_upload_folder(
    local_path: str,
    hdfs_path: str,
    obs_root_path: str,
    process_count: int,
    overwrite: bool,
    hdfs_user: str,
):
    print(
        "multi_process_upload_folder===================== start time is:"
        + format(datetime.now())
    )
    if process_count > os.cpu_count():
        raise ValueError(
            "The number of process_count must be less than the number of CPU cores"
        )

    # path clear
    hdfs_path = os.path.normpath(hdfs_path)
    local_path = os.path.normpath(local_path)

    # 确保本地目录存在
    if not os.path.exists(local_path):
        raise ValueError("local_path not exists.")

    # multiprocessing.set_start_method('spawn')
    set_start_multiprocess()
    local_base_name = os.path.basename(local_path)
    hdfs_full_path = os.path.join(hdfs_path, local_base_name)

    # if overwrite:
    process_futures = []
    with multiprocessing.Pool(processes=process_count) as pool:
        process_futures.append(
            pool.apply_async(
                process_if_overwrite,
                args=(hdfs_full_path, obs_root_path, overwrite, hdfs_user),
            )
        )
        for process_future in process_futures:
            process_future.get()

    with multiprocessing.Pool(processes=process_count) as pool:
        dir_futures = []
        for root, dirs, files in os.walk(local_path):
            hdfs_dirs = [
                os.path.join(
                    hdfs_path,
                    local_base_name,
                    root.replace(local_path, "").lstrip("/"),
                    local_dir,
                )
                for local_dir in dirs
            ]
            local_dirs = [os.path.join(root, local_dir) for local_dir in dirs]
            for i in range(len(hdfs_dirs)):
                dir_futures.append(
                    pool.apply_async(
                        process_dirs,
                        args=(local_dirs[i], hdfs_dirs[i], obs_root_path, hdfs_user),
                    )
                )
        # 等待所有任务完成
        for dir_future in dir_futures:
            dir_future.get()

    with multiprocessing.Pool(processes=process_count) as pool:
        file_futures = []
        for root, dirs, files in os.walk(local_path):
            hdfs_file_path = os.path.join(
                hdfs_path, local_base_name, root.replace(local_path, "").lstrip("/")
            )
            local_files = [os.path.join(root, local_file) for local_file in files]
            for i in range(len(local_files)):
                file_futures.append(
                    pool.apply_async(
                        process_files,
                        args=(
                            local_files[i],
                            hdfs_file_path,
                            obs_root_path,
                            hdfs_user,
                            overwrite,
                        ),
                    )
                )
        # 等待所有任务完成
        for file_future in file_futures:
            file_future.get()
    print(
        "multi_process_upload_folder===================== end time is:"
        + format(datetime.now())
    )


def process_download_folder(
    hdfs_path: str,
    hdfs_file_path: str,
    local_path: str,
    obs_root_path: str,
    overwrite: bool,
    hdfs_user: str,
):
    global insecure_client
    if not insecure_client:
        insecure_client = InsecureClient(obs_root_path, _user=hdfs_user)

    hdfs_relative_path = os.path.join(
        os.path.basename(hdfs_path), hdfs_file_path.replace(hdfs_path, "").lstrip("/")
    )
    local_file_path = os.path.join(local_path, hdfs_relative_path)
    local_direct_path = os.path.dirname(local_file_path)
    insecure_client.download_single_file(hdfs_file_path, local_direct_path, overwrite)


def multi_process_download_folder(
    hdfs_path: str,
    local_path: str,
    obs_root_path: str,
    process_count: int,
    overwrite: bool,
    hdfs_user: str,
):
    if process_count > os.cpu_count():
        raise ValueError(
            "The number of process_count must be less than the number of CPU cores"
        )

    # path clear
    hdfs_path = os.path.normpath(hdfs_path)
    local_path = os.path.normpath(local_path)

    # 确保本地目录存在
    local_path_file = os.path.normpath(local_path) + "/" + os.path.basename(hdfs_path)
    if not os.path.exists(local_path_file):
        os.makedirs(local_path_file)

    if overwrite:
        shutil.rmtree(local_path_file)

    # multiprocessing.set_start_method('spawn')
    set_start_multiprocess()
    with multiprocessing.Pool(processes=process_count) as pool:
        download_futures = []
        insecure_client_ins = InsecureClient(obs_root_path, _user=hdfs_user)
        # 遍历HDFS目录中的文件和子目录
        hdfs_files = list(insecure_client_ins.fs.walk(hdfs_path))
        for hdfs_file in tqdm(hdfs_files, desc="Gen download tasks"):
            hdfs_file_path = hdfs_file["path"].replace(obs_root_path, "/")
            if hdfs_file["kind"] == "directory":
                local_full_path = os.path.join(
                    local_path_file, hdfs_file_path.replace(hdfs_path, "").lstrip("/")
                )
                if not os.path.exists(local_full_path):
                    os.makedirs(local_full_path)
            else:
                download_futures.append(
                    pool.apply_async(
                        process_download_folder,
                        args=(
                            hdfs_path,
                            hdfs_file_path,
                            local_path,
                            obs_root_path,
                            overwrite,
                            hdfs_user,
                        ),
                    )
                )

        # 使用tqdm创建另一个进度条来跟踪任务完成情况
        for download_future in tqdm(download_futures, desc="Running download tasks"):
            download_future.get()


def getLogger():
    logger = logging.getLogger("demo")
    logger.setLevel(logging.INFO)
    return logger


def upload(
    local_path: str,
    hdfs_path: str,
    obs_root_path: str,
    process_count: int,
    overwrite: bool,
    hdfs_user: str,
):
    logger = getLogger()
    logger.info("Start to upload, overwrite is %s", overwrite)

    # path clear
    hdfs_path = os.path.normpath(hdfs_path)
    local_path = os.path.normpath(local_path)

    if not os.path.exists(local_path):
        logger.error("Local path not exists:" + local_path)
        return None

    if os.path.isfile(local_path):
        logger.info("Uploading file: %s ...", local_path)
        insecure_client_ins = InsecureClient(obs_root_path, _user=hdfs_user)
        # hdfs_path is not exists, create folder
        insecure_client_ins.create_folder(hdfs_path)
        insecure_client_ins.upload_single_file(local_path, hdfs_path, overwrite)
    else:
        logger.info("Uploading folder: %s ...", local_path)
        multi_process_upload_folder(
            local_path, hdfs_path, obs_root_path, process_count, overwrite, hdfs_user
        )
    logger.info("upload local path %s to hdfs path %s success.", local_path, hdfs_path)
    return None


def download(
    hdfs_path: str,
    local_path: str,
    obs_root_path: str,
    process_count: int,
    overwrite: bool,
    hdfs_user: str,
):
    logger = getLogger()
    logger.info("Start to download, overwrite is %s", overwrite)

    # path clear
    hdfs_path = os.path.normpath(hdfs_path)
    local_path = os.path.normpath(local_path)

    if os.path.isfile(local_path):
        logger.error("download local_path must dir,please check : %s ...", local_path)
        raise RuntimeError("download local_path must dir,please check : %s ...", local_path)

    insecure_client_ins = InsecureClient(obs_root_path, _user=hdfs_user)

    if not insecure_client_ins.hdfs_path_exists(hdfs_path):
        logger.error(
            "download hdfs_path is not exists,please check : %s ...", hdfs_path
        )
        raise RuntimeError("download hdfs_path is not exists,please check : %s ...", hdfs_path)

    if insecure_client_ins.hdfs_path_isfile(hdfs_path):

        # 本地路径不存在，创建本地目录
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        insecure_client_ins.download_single_file(hdfs_path, local_path, overwrite)
        logger.info("Downloading file: %s to %s...", hdfs_path, local_path)
    else:
        logger.info("Downloading folder: %s to %s...", hdfs_path, local_path)
        multi_process_download_folder(
            hdfs_path, local_path, obs_root_path, process_count, overwrite, hdfs_user
        )
