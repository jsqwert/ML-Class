# -- encoding:utf-8 --
"""
Create by ibf on 2018/6/21
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 随机数据产生
# 给定随机数种子：当程序多次运行的时候，可以保证每次运行时候随机的数据都是一样的
np.random.seed(28)
n = 100
b_values = np.random.normal(loc=-1.0, scale=10.0, size=n)
c_values = np.random.normal(loc=0.0, scale=1.0, size=n)
print("b的均值:{}".format(np.mean(b_values)))

# # 随机数据可视化查看
# plt.figure(facecolor='w')
# plt.subplot(1, 2, 1)
# plt.hist(b_values, 1000, color='#FF0000')
# plt.subplot(1, 2, 2)
# plt.hist(c_values, 1000, color='#00FF00')
# plt.suptitle(u'随机数据可视化', fontsize=22)
# plt.show()

def calc_min_value_with_one_sample(b_values, c_values, max_iter=1000, tol=0.00001, alpha=0.01):
    """
    计算最小值时候对应的x和y的值
    :param b_values: 样本对应的b值
    :param c_values: 样本对应的c值
    :param max_iter: 最大迭代次数
    :param tol: 当变量小于该值的时候收敛
    :param alpha: 梯度下降学习率
    :return:
    """

    def f(x, b, c):
        """
        原始函数
        :param x:
        :param b:
        :param c:
        :return:
        """
        return x ** 2 + b * x + c

    def h(x, b, c):
        """
        原始函数对应的导函数
        :param x:
        :param b:
        :param c:
        :return:
        """
        return 2 * x + b

        # 定义变量

    step_channge = 1.0 + tol
    step = 0

    # 获取第一个样本
    b = b_values[0]
    c = c_values[0]

    # 给定一个初始的x值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b, c)

    print("当前参数为:")
    print("b={}".format(b))
    print("c={}".format(c))

    # 开始迭代循环
    while step_channge > tol and step < max_iter:
        # 1. 计算梯度值
        current_d_f = h(current_x, b, c)
        # 2. 更新参数
        current_x = current_x - alpha * current_d_f
        # 3. 计算更新x之后的y值
        tmp_y = f(current_x, b, c)
        # 4. 记录y的变换大小、更新迭代次数、更新当前的y值
        step_channge = np.abs(current_y - tmp_y)
        step += 1
        current_y = tmp_y
    print("最终更新的次数:{}, 最终的变化率:{}".format(step, step_channge))
    print("最终结果为:{}---->{}".format(current_x, current_y))


def calc_min_value_with_ten_sample(n, b_values, c_values, max_iter=1000, tol=0.00001, alpha=0.01):
    """
    计算最小值时候对应的x和y的值
    :param n: 样本数量
    :param b_values: 样本对应的b值
    :param c_values: 样本对应的c值
    :param max_iter: 最大迭代次数
    :param tol: 当变量小于该值的时候收敛
    :param alpha: 梯度下降学习率
    :return:
    """
    # 要求n必须等于10
    assert n == 10 and len(b_values) == n and len(c_values) == n

    def f(x, b_values, c_values):
        """
        原始函数
        :param x:
        :param b_values:
        :param c_values:
        :return:
        """
        sample_1 = x ** 2 + b_values[0] * x + c_values[0]
        sample_2 = x ** 2 + b_values[1] * x + c_values[1]
        sample_3 = x ** 2 + b_values[2] * x + c_values[2]
        sample_4 = x ** 2 + b_values[3] * x + c_values[3]
        sample_5 = x ** 2 + b_values[4] * x + c_values[4]
        sample_6 = x ** 2 + b_values[5] * x + c_values[5]
        sample_7 = x ** 2 + b_values[6] * x + c_values[6]
        sample_8 = x ** 2 + b_values[7] * x + c_values[7]
        sample_9 = x ** 2 + b_values[8] * x + c_values[8]
        sample_10 = x ** 2 + b_values[9] * x + c_values[9]
        return sample_1 + sample_2 + sample_3 + sample_4 + sample_5 + sample_6 + sample_7 + sample_8 + sample_9 + sample_10

    def h(x, b_values, c_values):
        """
        原始函数对应的导函数
        :param x:
        :param b_values:
        :param c_values:
        :return:
        """
        sample_1 = x * 2 + b_values[0]
        sample_2 = x * 2 + b_values[1]
        sample_3 = x * 2 + b_values[2]
        sample_4 = x * 2 + b_values[3]
        sample_5 = x * 2 + b_values[4]
        sample_6 = x * 2 + b_values[5]
        sample_7 = x * 2 + b_values[6]
        sample_8 = x * 2 + b_values[7]
        sample_9 = x * 2 + b_values[8]
        sample_10 = x * 2 + b_values[9]
        return sample_1 + sample_2 + sample_3 + sample_4 + sample_5 + sample_6 + sample_7 + sample_8 + sample_9 + sample_10

    # 定义变量
    step_channge = 1.0 + tol
    step = 0

    # 给定一个初始的x值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b_values, c_values)

    print("当前参数为:")
    print("b_values={},b的均值为:{}".format(b_values, np.mean(b_values)))
    print("c_values={},c的均值为:{}".format(c_values, np.mean(c_values)))

    # 开始迭代循环
    while step_channge > tol and step < max_iter:
        # 1. 计算梯度值
        current_d_f = h(current_x, b_values, c_values)
        # 2. 更新参数
        current_x = current_x - alpha * current_d_f
        # 3. 计算更新x之后的y值
        tmp_y = f(current_x, b_values, c_values)
        # 4. 记录y的变换大小、更新迭代次数、更新当前的y值
        step_channge = np.abs(current_y - tmp_y)
        step += 1
        current_y = tmp_y
    print("最终更新的次数:{}, 最终的变化率:{}".format(step, step_channge))
    print("最终结果为:{}---->{}".format(current_x, current_y))


def calc_min_value_with_n_sample(n, b_values, c_values, max_iter=1000, tol=0.00001, alpha=0.01):
    """
    计算最小值时候对应的x和y的值
    :param n: 样本数量
    :param b_values: 样本对应的b值
    :param c_values: 样本对应的c值
    :param max_iter: 最大迭代次数
    :param tol: 当变量小于该值的时候收敛
    :param alpha: 梯度下降学习率
    :return:
    """

    def f1(x, b, c):
        return x ** 2 + b * x + c

    def f(x, b_values, c_values):
        """
        原始函数
        :param x:
        :param b_values:
        :param c_values:
        :return:
        """
        result = 0
        for b, c in zip(b_values, c_values):
            # 遍历所有b和c的组合，这里求均值(防止数据量太大，计算困难)
            result += f1(x, b, c) / n
        return result

    def h1(x, b, c):
        return x * 2 + b

    def h(x, b_values, c_values):
        """
        原始函数对应的导函数
        :param x:
        :param b_values:
        :param c_values:
        :return:
        """
        result = 0
        for b, c in zip(b_values, c_values):
            # 遍历求解每个b、c组合对应的梯度值，这里求均值(防止数据量太大，计算困难)
            result += h1(x, b, c) / n
        return result

    # 定义变量
    step_channge = 1.0 + tol
    step = 0

    # 给定一个初始的x值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b_values, c_values)

    print("当前参数为:")
    print("b_values={},b的均值为:{}".format(b_values, np.mean(b_values)))
    print("c_values={},c的均值为:{}".format(c_values, np.mean(c_values)))

    # 开始迭代循环
    while step_channge > tol and step < max_iter:
        # 1. 计算梯度值
        current_d_f = h(current_x, b_values, c_values)
        # 2. 更新参数
        current_x = current_x - alpha * current_d_f
        # 3. 计算更新x之后的y值
        tmp_y = f(current_x, b_values, c_values)
        # 4. 记录y的变换大小、更新迭代次数、更新当前的y值
        step_channge = np.abs(current_y - tmp_y)
        step += 1
        current_y = tmp_y
    print("最终更新的次数:{}, 最终的变化率:{}".format(step, step_channge))
    print("最终结果为:{}---->{}".format(current_x, current_y))


# print("*" * 50)
# calc_min_value_with_one_sample(b_values, c_values)
# print("*" * 50)
# calc_min_value_with_ten_sample(n, b_values, c_values)
# print("*" * 50)
# calc_min_value_with_n_sample(n, b_values, c_values)
