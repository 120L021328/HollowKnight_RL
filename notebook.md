# RL AI Knight

### Abstract

1. 每次保存下来的5个.pt文件的作用
   - bestonline 大于100且是10的倍数的训练中，进行测试，选效果最好的
   - besttrainonline 在训练过程中选效果最好的
   - latestonline 最后一次训练结果，在训练中中途退出也是会保留的，下面两个不会
   - latestoptimizer 最后一次训练的optimizer结果
   - latesttarget0 最后一次训练的target_model结果
2. Trainer中，有属性model、target_model、optimizer
   - model 继承自nn.Module的类，有state_dict存参数
   - target_model 设置target个数，可以实现Averaged-DQN，默认是1就无用
   - optimizer torch.optim.NAdam
3. 测试与训练的区别：noise的设定，测试时无

### ISSUE

1. 图像怎么输入的，貌似是取屏幕的固定位置的像素？
2. LOSS的断层式增大，有可能是因为选择了新的best model？

### TODO

1. 改reward函数，删除时间的奖励
2. 增加对于胜利次数和分布的显示
3. 增加best更新时，记录下更新位置

### LOG

1. 388 效果最好的一次，550次迭代、使用epsilon decay函数为100 / steps
2. 970 改为1000次迭代，函数也改为1000 / steps
3. 