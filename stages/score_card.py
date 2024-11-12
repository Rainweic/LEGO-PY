import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, average_precision_score
from pyecharts import options as opts
from pyecharts.charts import Line
from dags.stage import CustomStage

class CustomLogisticRegression(LogisticRegression):
    """扩展LogisticRegression以记录训练过程"""
    def __init__(
        self,
        penalty='l2',
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver='lbfgs',
        max_iter=100,
        multi_class='auto',
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )
        self.loss_history = []
        
    def _callback(self, params):
        """每次迭代后的回调函数"""
        if not hasattr(self, 'X_fit_'):
            return False
            
        # 计算当前参数下的预测概率
        z = -(self.X_fit_ @ params.reshape(-1, 1) + self.intercept_)
        proba = 1 / (1 + np.exp(z)).reshape(-1)
        
        # 计算并记录loss
        current_loss = log_loss(self.y_fit_, proba)
        self.loss_history.append(current_loss)
        return False

    def fit(self, X, y, sample_weight=None):
        """重写fit方法以记录训练过程"""
        self.loss_history = []  # 重置历史记录
        self.X_fit_ = X
        self.y_fit_ = y
        
        # 调用父类的fit方法
        super().fit(X, y, sample_weight)
        
        # 记录最终的loss
        if len(self.loss_history) == 0:
            z = -(X @ self.coef_.T + self.intercept_)
            final_proba = 1 / (1 + np.exp(z)).reshape(-1)
            final_loss = log_loss(y, final_proba)
            self.loss_history.append(final_loss)
        
        return self

    def _fit_liblinear(self, X, y, sample_weight):
        """liblinear solver不支持callback，需要特殊处理"""
        super()._fit_liblinear(X, y, sample_weight)
        # 至少记录最终的loss
        z = -(X @ self.coef_.T + self.intercept_)
        final_proba = 1 / (1 + np.exp(z)).reshape(-1)
        self.loss_history = [log_loss(y, final_proba)]
        return self

    def _fit_lbfgs(self, X, y, sample_weight):
        """lbfgs solver的特殊处理"""
        self._solver_options = {
            'callback': self._callback,
            'maxiter': self.max_iter
        }
        return super()._fit_lbfgs(X, y, sample_weight)

class ScoreCard(CustomStage):
    def __init__(self, features, label_col, train_params=None, base_score=600, pdo=20, base_odds=50):
        """初始化评分卡模型
        
        Args:
            features (list): 特征列表
            label (str): 标签列名
            train_params (dict): 逻辑回归训练参数
                - C: 正则化强度的倒数，越小正则化越强
                - class_weight: 类别权重，处理样本不平衡
                - max_iter: 最大迭代次数
                - random_state: 随机种子
            base_score (int): 基础分，通常设置为600或500
            pdo (int): Points to Double the Odds，通常设置为20或40
            base_odds (float): 基础分对应的好坏比，通常设置为50或20
        """
        super().__init__(n_outputs=1)
        self.features = features
        self.label = label_col
        self.model = None
        self.metrics = {}
        
        # 逻辑回归参数
        self.train_params = train_params or {
            'C': 1.0,  # 正则化强度，可选值范围：[0.001, 0.01, 0.1, 1.0, 10.0]
            'class_weight': 'balanced',  # 可选值：None, 'balanced', {0:w0, 1:w1}
            'max_iter': 1000,  # 通常500-2000足够
            'random_state': 42,
            'solver': 'lbfgs',  # 推荐使用'lbfgs'或'newton-cg'
            'tol': 1e-4  # 收敛容差
        }
        
        # 评分卡参数
        self.base_score = base_score  # 基础分
        self.pdo = pdo  # 翻倍分数
        self.base_odds = base_odds  # 基础好坏比
        
    def _calculate_metrics(self, y_true, y_pred_proba):
        """计算模型评估指标"""
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'avg_precision': average_precision_score(y_true, y_pred_proba)
        }
        
        # 计算KS值
        fpr, tpr, _ = precision_recall_curve(y_true, y_pred_proba)
        ks = max(np.abs(tpr - fpr))
        metrics['ks'] = ks
        
        return metrics
        
    def _calculate_score_params(self):
        """计算评分卡参数"""
        B = self.pdo / np.log(2)
        A = self.base_score - B * np.log(self.base_odds)
        return A, B
        
    def _sample_loss_history(self, loss_history, max_points=100):
        """对loss历史进行均匀采样
        
        Args:
            loss_history (list): 原始loss历史数据
            max_points (int): 最大采样点数
            
        Returns:
            tuple: (采样后的索引列表, 采样后的loss值列表)
        """
        n_points = len(loss_history)
        if n_points <= max_points:
            return list(range(1, n_points + 1)), loss_history
            
        # 计算采样间隔
        step = n_points / max_points
        
        # 生成采样索引
        indices = [int(i * step) for i in range(max_points)]
        indices[-1] = min(indices[-1], n_points - 1)  # 确保不越界
        
        # 采样loss值
        sampled_loss = [loss_history[i] for i in indices]
        
        # 生成对应的x轴标签
        x_labels = [i + 1 for i in indices]
        
        # 确保包含第一个和最后一个点
        if indices[0] != 0:
            x_labels.insert(0, 1)
            sampled_loss.insert(0, loss_history[0])
        if indices[-1] != n_points - 1:
            x_labels.append(n_points)
            sampled_loss.append(loss_history[-1])
            
        return x_labels, sampled_loss

    def _plot_loss_history(self, loss_history, max_points=100):
        """绘制loss历史曲线
        
        Args:
            loss_history (list): loss历史数据
            max_points (int): 最大显示点数
        """
        # 对数据进行采样
        x_labels, sampled_loss = self._sample_loss_history(loss_history, max_points)
        
        self.logger.info(f"Loss历史数据点数: 原始={len(loss_history)}, 采样后={len(sampled_loss)}")
        
        line = (
            Line()
            .add_xaxis(x_labels)
            .add_yaxis(
                "LogLoss",
                sampled_loss,
                is_smooth=True,  # 使用平滑曲线
                symbol_size=6,   # 数据点大小
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="训练过程中的LogLoss变化",
                    subtitle=f"(采样显示{len(sampled_loss)}个点)"
                ),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    name="迭代次数",
                    name_location="center",
                    name_gap=30,
                    splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                yaxis_opts=opts.AxisOpts(
                    name="LogLoss",
                    name_location="center",
                    name_gap=40,
                    splitline_opts=opts.SplitLineOpts(is_show=True)
                )
            )
        )
        return {"LogLoss": line.dump_options_with_quotes()}

    @staticmethod
    def predict(model, data: pl.LazyFrame) -> pl.LazyFrame:
        """静态预测方法
        
        Args:
            model: 已训练的模型对象（包含LR模型和评分卡参数）
            data: 待预测数据
            
        Returns:
            包含预测概率和分数的LazyFrame
        """
        # 转换为numpy数组
        X = data.select(model['features']).collect().to_numpy()

        proba = 1 / (1 + np.exp(-(X @ np.array(model['weight']).T + model['bias']))).reshape(-1)

        # 计算分数
        odds = proba / (1 - proba)
        A, B = model['score_params']
        scores = A + B * np.log(odds)
        
        # 添加预测结果
        result = data.select([
            pl.col("*"),
            pl.Series("y_score", proba),
            pl.Series("score", scores.round().astype(int))
        ])

        return result

    def forward(self, train_woe: pl.LazyFrame, eval_woe: pl.LazyFrame=None):
        """训练评分卡模型"""
        # 准备训练数据
        X_train = train_woe.select(self.features).collect().to_numpy()
        y_train = train_woe.select(self.label).collect().to_numpy().ravel()
        
        # 准备评估数据
        if eval_woe is not None:
            X_eval = eval_woe.select(self.features).collect().to_numpy()
            y_eval = eval_woe.select(self.label).collect().to_numpy().ravel()
        
        # 用自定义的逻辑回归模型
        lr = CustomLogisticRegression(**self.train_params)
        lr.fit(X_train, y_train)
        
        # 计算训练集指标
        train_proba = lr.predict_proba(X_train)[:, 1]
        self.metrics['train'] = self._calculate_metrics(y_train, train_proba)
        
        # 计算评估集指标
        if eval_woe is not None:
            eval_proba = lr.predict_proba(X_eval)[:, 1]
            self.metrics['eval'] = self._calculate_metrics(y_eval, eval_proba)
        
        # 计算评分卡参数
        A, B = self._calculate_score_params()
        
        # 保存模型和相关参数
        model_info = {
            'type': 'ScoreCard',
            'weight': lr.coef_.tolist(),
            'bias': float(lr.intercept_),
            'features': self.features,
            'score_params': (float(A), float(B)),
        }
        
        # # 使用静态predict方法生成预测结果
        # result = self.predict(model_info, eval_woe)
        
        # 生成loss历史图表
        self.summary.append(self._plot_loss_history(lr.loss_history))
        
        # 记录summary信息
        train_info = {
            "训练集指标": {
                "AUC": f"{self.metrics['train']['auc']:.4f}",
                "KS": f"{self.metrics['train']['ks']:.4f}",
                "LogLoss": f"{self.metrics['train']['log_loss']:.4f}",
                "AvgPrecision": f"{self.metrics['train']['avg_precision']:.4f}"
            },
            "模型参数": {
                "基础分": self.base_score,
                "PDO": self.pdo,
                "基础好坏比": self.base_odds,
                "特征数量": len(self.features),
                "正则化系数": self.train_params.get('C', 1.0),
                "迭代次数": len(lr.loss_history)
            }
        }

        if eval_woe is not None:
            train_info["评估集指标"] = {
                "AUC": f"{self.metrics['eval']['auc']:.4f}",
                "KS": f"{self.metrics['eval']['ks']:.4f}",
                "LogLoss": f"{self.metrics['eval']['log_loss']:.4f}",
                "AvgPrecision": f"{self.metrics['eval']['avg_precision']:.4f}"
            },

        self.logger.info(train_info)
        
        return model_info