import dgl
import torch
import argparse
graph = dgl.load_graphs('result/大模型/thg.dgl')[0][0]

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='大模型', help='数据集')

parser.add_argument('--log_dir', type=str, default='log', help='日志保存路径')
parser.add_argument('--save_dir', type=str, default='output', help='模型输出向量保存路径')
parser.add_argument('--csv_dir', type=str, default='csv', help='模型评估结果保存路径')
parser.add_argument('--model_dir', type=str, default='model', help='模型保存路径')

parser.add_argument('--GPU_ID', type=str, default='0', help='GPU id')

parser.add_argument('--graph_path', type=str, default="D:/Hu/技术机会预测/result/大模型/thg.dgl", help='网络保存地址(绝对路径)')
# parser.add_argument('--graph_path', type=str, default="D:/Hu/技术机会预测/result/大模型/thg1.dgl", help='网络保存地址(绝对路径)')
parser.add_argument('--graphs_path', type=str, default="D:/Hu/技术机会预测/result/大模型/target_graphs.dgl", help='网络snapshots保存地址(绝对路径)')
# parser.add_argument('--metapath_path', type=str, default="D:/Hu/技术机会预测/result/大模型/temporal_metapath_dict.pkl", help='metapath字典保存地址(绝对路径)')
parser.add_argument('--metapath_path', type=str, default="D:/Hu/技术机会预测/result/大模型/temporal_metapath_dict_paper1.pkl", help='metapath字典保存地址(绝对路径)')
parser.add_argument('--time_range', type=list, default=list(range(2010, 2023)), help='训练数据时间范围')
parser.add_argument('--test_timestamp', type=int, default=2023, help='测试时间步')
parser.add_argument('--min_timestamp', type=int, default=2010, help='最小timestamp')
parser.add_argument('--max_timestamp', type=int, default=2022, help='最大timestamp')
parser.add_argument('--num_metapath_types', type=int, default=3, help='metapath种类数')

parser.add_argument('--device', default=torch.device('cuda:0'), help='训练设备')
# parser.add_argument('--device', default='cpu', help='训练设备')
parser.add_argument('--num_epochs', type=int, default=20, help='训练轮次')
parser.add_argument('--batch_size', type=int, default=128, help='批大小')
parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
parser.add_argument('--patience', type=int, default=5, help='early stop')

parser.add_argument('--feature_dim', type=int, default=768, help='特征向量维度')
parser.add_argument('--hidden_dim', type=int, default=64, help='隐向量维度', choices=[16, 32, 64, 128, 256]) # best=128
parser.add_argument('--output_dim', type=int, default=64, help='输出向量维度', choices=[16, 32, 64, 128, 256]) # best=64
parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout概率')
parser.add_argument('--num_heads', type=int, default=8, help='多头注意力机制', choices=[4, 6, 8, 12, 16]) # best=8
parser.add_argument('--num_logits_heads', type=int, default=8, help='talking-heads', choices=[4, 6, 8, 12, 16])

parser.add_argument('--num_ipc', type=int, default=graph.num_nodes(ntype='ipc'), help='ipc节点数')
parser.add_argument('--num_patent', type=int, default=graph.num_nodes(ntype='patent'), help='patent节点数')
parser.add_argument('--num_paper', type=int, default=graph.num_nodes(ntype='paper'), help='paper节点数')

args = parser.parse_args()