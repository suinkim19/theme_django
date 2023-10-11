from django.apps import AppConfig
from django_plotly_dash import DjangoDash
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objs as go
import copy
import random

# 데이터프레임 불러오기
replace_dict = {True: '#f7f7f7', False: '#F8CBAD'}

def load_weekly_data(week):
    
    # 데이터 파일 이름 지정
    node_file = 'network/result/nodes_' + week + '.csv'
    edge_file = 'network/result/edges_' + week + '.csv'
    theme_index_file = 'network/result/theme_index_' + week + '.csv'
    co_imb_file = 'network/result/imb_themed_stock_' + week + '.pickle'
    
    # 데이터 불러오기
    node_df = pd.read_csv(node_file)
    edge_df = pd.read_csv(edge_file)
    theme_index = pd.read_csv(theme_index_file)
    with open(co_imb_file, 'rb') as fr:
              co_imb_themed_stock = pickle.load(fr)

    # 인덱스 컬럼이 없을 시 생성
    if 'index' not in node_df.columns:
        node_df['index'] = node_df.index

    if 'node1_index' not in edge_df.columns:
        # 'node1' 열에 대한 'node1_index' 열 추가
        edge_df['node1_index'] = edge_df['node1'].map(node_df.set_index('node')['index'])

        # 'node2' 열에 대한 'node2_index' 열 추가
        edge_df['node2_index'] = edge_df['node2'].map(node_df.set_index('node')['index'])

        
    theme_stock_df = pd.DataFrame(co_imb_themed_stock.items(), columns=['node', 'column'])
        
        # 마지막 숫자 항목 추출
    theme_stock_df['similarity'] = theme_stock_df['column'].apply(lambda x: x[-1])

    # 각 튜플의 첫 번째 값 추출
    theme_stock_df['stock'] = theme_stock_df['column'].apply(lambda x: [item[0] for item in x[:-1]])

    # 각 튜플의 두 번째 값 추출
    theme_stock_df['TF'] = theme_stock_df['column'].apply(lambda x: [item[1] for item in x[:-1]]).apply(lambda x: [replace_dict.get(item, item) for item in x])

    # 원래의 'column' 컬럼 제거
    theme_stock_df = theme_stock_df.drop(columns=['column'])

    node_df = pd.merge(node_df, theme_stock_df, on='node', how='left')

    # 클러스터 정보가 들어있는 컬럼명
    cluster_option = 'cluster'

    # 클러스터 개수 확인
    num_clusters = node_df[cluster_option].nunique()

    # 클러스터 색상
    cluster_colors = {
        cluster: f'hsl({int(360 * i / num_clusters)}, 70%, 50%)'
        for i, cluster in enumerate(node_df[cluster_option].unique())
    }

    # Prepare data
    nodes = [
        {
            'data': {
                'id': str(row['index']),
                'label': row['node'],
                'cluster': row[cluster_option],
                'parent': 'cluster_' + str(row[cluster_option])
            },
            'grabbable': True,
            'style': {
                'background-color': cluster_colors[row[cluster_option]],
                'width': row['count_rescaled']*5,
                'height': row['count_rescaled']*5,
                'font-family': 'The jamsil ExtraBold',
                'font-size': 10 + 0.5 * row['count_rescaled'],  # 글씨 크기 조절
                'text-valign': 'center',  # 세로 중앙 정렬
                'text-halign': 'center',  # 가로 중앙 정렬
                'background-opacity': 0.7,
            }
        }
        for _, row in node_df.iterrows()
    ]

    edges = [
        {
            'data': {
                'source': str(row['node1_index']),
                'target': str(row['node2_index']),
                'weight': row['weight']
            },
            'style': {
                'line-color': 'rgba(200, 200, 200, 0.2)',
                'width': row['weight_rescaled'] * 2
            }
        }
        for _, row in edge_df.iterrows()
    ]

    return node_df, edge_df, theme_index, nodes, edges

node_df, edge_df, theme_index, nodes, edges = load_weekly_data('june_week2')

################################################################## 필요 함수 부분
def get_node_info(node_index, node_df, theme_index):
    summary = str(node_df[node_df['index'] == node_index]['AI_summary'].iloc[0])
    stocks = node_df[node_df['index'] == node_index]['stock'].iloc[0]
    stock_similarity = node_df[node_df['index'] == node_index]['similarity'].iloc[0]
    stock_color = node_df[node_df['index'] == node_index]['TF'].iloc[0]
    cluster_name = node_df[node_df['index'] == node_index]['cluster_name'].iloc[0]
    dates = theme_index['일자'].tolist()
    node_label = node_df[node_df['index'] == node_index]['node'].iloc[0]
    values = theme_index[node_label].tolist()

    return summary, stocks, stock_similarity, stock_color, cluster_name, dates, node_label, values

def create_figure(dates, values, node_label):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name=node_label, line=dict(color='#164C8A')))
    
    figure.update_layout(
                font=dict(family="The jamsil Bold, sans-serif"),
                xaxis_title="일자",
                yaxis_title="테마 지수",
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',  # 투명한 배경
                xaxis=dict(
                    showgrid=False,  # 그리드 숨기기
                    zeroline=False,
                    showline=True,
                    linecolor='black'  # 검정색 x축
                ),
                yaxis=dict(
                    showgrid=False,  # 그리드 숨기기
                    zeroline=False,
                    showline=True,
                    linecolor='black',  # 검정색 y축
                    tickformat=',.0f'  # y축 단위 변경 없음
                ),
                hoverlabel=dict(
                    font=dict(
                        family="nanum-gothic, sans-serif")
                )
            )
    
    return figure

def create_content_div(node_label, cluster_name, summary, stocks, stock_similarity, stock_color, figure):
    if pd.isna(stock_similarity):
        stock_sim ='-'
    else:
        stock_sim = round(stock_similarity, 4)
    return html.Div(
                [
                    html.P(node_label + ' - 소속 클러스터 : ' + cluster_name,
                        style={'font-family': 'The jamsil ExtraBold', 'font-size': '25px'}),
                    html.H5("AI의 요약:", style={'font-family': 'The jamsil Bold', 'font-size': '20px'}),
                    html.P(summary, style={'font-family': 'nanum-gothic', 'font-size': '20px'}),
                    html.H5("관련 주식 종목:", style={'font-family': 'The jamsil Bold', 'font-size': '20px'}),
                    html.Div(
                        [
                            html.Span(f"#{stock}", style={
                                'border': '1px solid #ddd',
                                'font-family': 'The jamsil Bold',
                                'padding': '5px 10px',
                                'margin': '2px',
                                'border-radius': '5px',
                                'background-color':  stock_color[idx],
                                'display': 'inline-block'
                            })
                            for idx, stock in enumerate(stocks)
                        ] + 
                        [
                            html.P(f"평균 유사도: {stock_sim}", style={'font-family': 'nanum-gothic', 'font-size': '20px'})
                        ]
                    ),
                    html.H5("테마 지수:", style={'font-family': 'The jamsil Bold', 'font-size': '20px'}),
                    html.Div(dcc.Graph(figure=figure))
                ],
                style={
                    'border': '2px solid #ddd',               # 테두리 설정
                    'border-radius': '15px',                  # 둥근 테두리 설정
                    'box-shadow': '5px 5px 10px rgba(0,0,0,0.1)',  # 그림자 효과 설정
                    'padding': '50px',                        # 안쪽 여백 설정
                    'background-color': 'white',              # 배경색 설정
                    'width': '90%',                           # div의 너비 설정
                    'margin': '20px auto'                     # 중앙 정렬
                }
            )

#######################################################

# Dash app
app = DjangoDash('dash_app_june_week2', add_bootstrap_links=True, external_stylesheets=["/static/css/font.css"])

# App layout
app.layout = html.Div([
    html.Div([
        html.Label("테마 검색:", style = {'font-family': 'The jamsil Bold'}),  # 검색창 라벨
        dcc.Input(id='search-input', type='text', placeholder='테마를 입력하세요...', n_submit=0),  # n_submit 추가
        html.Button(id='search-button', n_clicks=0, children='검색',style = {'font-family': 'The jamsil Bold'}),  # 검색 버튼
        html.Div(id='search-output'),  # 검색 결과 출력을 위한 Div
    ], style={'width': '100%', 'textAlign': 'center', 'padding': '20px'}),
    html.Div([
        cyto.Cytoscape(
            id='cytoscape-network',
            layout={'name': 'cose'},
            style={'width': '100%', 'height': '700px'},
            minZoom=0.1,
            maxZoom=4.0,
            elements = nodes + edges   # 전체 네트워크를 기본으로 설정
        )
    ], style={'width': '50%', 'display': 'inline-block'}),
    html.Div(id='cluster-name', style={
        'position': 'absolute',
        'top': '5px',
        'right': '5px',
        'background-color': 'rgba(255,255,255,0.8)',
        'padding': '5px',
        'border-radius': '5px',
        'font-family': 'The jamsil ExtraBold',
        'font-size': '30px',
        'z-index': '1000'  # Ensure it's on top
    }),
    html.Div([
        cyto.Cytoscape(
            id='cytoscape-cluster',
            layout={'name': 'cose'},
            style={'width': '100%', 'height': '700px'},
            minZoom=0.1,
            maxZoom=4.0,
            elements=[]  # 초기에는 비어있는 네트워크를 표시
        )
    ], style={'width': '50%', 'display': 'inline-block'}),
    html.Div(id='explanation')
])





@app.callback(
    [Output('cytoscape-cluster', 'elements'), 
    Output('explanation', 'children'), 
    Output('cluster-name', 'children')],
    [Input('cytoscape-network', 'tapNodeData'),
    Input('cytoscape-cluster', 'tapNodeData'),
    Input('search-button', 'n_clicks'),
    Input('search-input', 'n_submit')],
    [State('search-input', 'value')]
)


def display_cluster_network(main_node_data, cluster_node_data, search_n_clicks, search_n_submit, search_value, week = 'june_week2'):
    
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if (input_id == 'search-button' or input_id == 'search-input') and search_value:  # 검색 버튼이 클릭되었고, 검색 값이 있는 경우
        # search_value를 사용해 node_df에서 노드를 찾아서 해당 노드의 정보와 연관된 내용을 반환

        matched_nodes = node_df[node_df['node'].str.contains(search_value, case=False)]  # 대소문자 구분 없이 노드 찾기

        if not matched_nodes.empty:
            first_node = matched_nodes.iloc[0]
            node_index = first_node['index']
            summary, stocks, stock_similarity, stock_color, cluster_name, dates, node_label, values = get_node_info(node_index, node_df, theme_index)

            figure = create_figure(dates, values, node_label)
            content_div = create_content_div(node_label, cluster_name, summary, stocks, stock_similarity, stock_color, figure)
            nodes_copy = copy.deepcopy(nodes)
            edges_copy = copy.deepcopy(edges)

            cluster = first_node['cluster']
            cluster_nodes = [ele for ele in nodes_copy if ele['data']['cluster'] == cluster]
            cluster_node_ids = [node['data']['id'] for node in cluster_nodes]
            cluster_edges = [ele for ele in edges_copy if ele['data']['source'] in cluster_node_ids and ele['data']['target'] in cluster_node_ids]

            clicked_node_id = str(node_index).replace('_clicked','').replace('_connected','')
            node_style = cluster_nodes[0]['style']['background-color']

            connected_edges = [edge for edge in cluster_edges if edge['data']['source'] == clicked_node_id or edge['data']['target'] == clicked_node_id]

            connected_node_ids = set()
            for edge in connected_edges:
                connected_node_ids.add(edge['data']['source'])
                connected_node_ids.add(edge['data']['target'])
            for edge in cluster_edges:
                if edge['data']['source'] in connected_node_ids:
                    if edge in connected_edges:
                        edge['style']['line-color'] = node_style

                    if edge['data']['source'] == clicked_node_id:
                        edge['data']['source'] += '_clicked'
                    else:
                        edge['data']['source'] += '_connected'
                    print(edge)
                if edge['data']['target'] in connected_node_ids:
                    if edge in connected_edges:
                        edge['style']['line-color'] = node_style
                    if edge['data']['target'] == clicked_node_id:
                        edge['data']['target'] += '_clicked'
                    else:
                        edge['data']['target'] += '_connected'
                    print(edge)

            for node in cluster_nodes:
                if node['data']['id'] in connected_node_ids:
                    node['style'].pop('border-color', None)
                    if node['data']['id'] not in [clicked_node_id,clicked_node_id + '_clicked']:
                        node['data']['id'] += '_connected'
                    node['style']['background-opacity'] = 0.8
                else:
                    node['style']['background-opacity'] = 0.6

                if node['data']['id'] in [clicked_node_id,clicked_node_id + '_clicked']:
                    node['style']['border-opacity'] = 0.7
                    node['data']['id'] += '_clicked'
                    node['style']['border-color'] = 'gold'
                    node['style']['border-width'] = 15
                    node['style']['background-opacity'] = 0.95
                    
            return cluster_nodes + cluster_edges, content_div, cluster_name
        else:
            return [], "검색 결과가 없습니다.", ""

    elif input_id in ['cytoscape-network', 'cytoscape-cluster']:

        node_data = main_node_data if input_id == 'cytoscape-network' else cluster_node_data

        if node_data:
            node_index = int(node_data['id'].replace('_clicked','').replace('_connected',''))
            summary, stocks, stock_similarity, stock_color,  cluster_name, dates, node_label, values = get_node_info(node_index, node_df, theme_index)

            figure = create_figure(dates, values, node_label)
            content_div = create_content_div(node_label, cluster_name, summary, stocks, stock_similarity, stock_color,  figure)
            
            clicked_node_id = node_data['id'].replace('_clicked','').replace('_connected','')
            
            nodes_copy = copy.deepcopy(nodes)
            edges_copy = copy.deepcopy(edges)

            cluster = node_data['cluster']
            cluster_nodes = [ele for ele in nodes_copy if ele['data']['cluster'] == cluster]
            cluster_node_ids = [node['data']['id'] for node in cluster_nodes]
            cluster_edges = [ele for ele in edges_copy if ele['data']['source'] in cluster_node_ids and ele['data']['target'] in cluster_node_ids]

            node_style = cluster_nodes[0]['style']['background-color']

            connected_edges = [edge for edge in cluster_edges if edge['data']['source'] == clicked_node_id or edge['data']['target'] == clicked_node_id]

            connected_node_ids = set()
            for edge in connected_edges:
                connected_node_ids.add(edge['data']['source'])
                connected_node_ids.add(edge['data']['target'])
            for edge in cluster_edges:
                if edge['data']['source'] in connected_node_ids:
                    if edge in connected_edges:
                        edge['style']['line-color'] = node_style

                    if edge['data']['source'] == clicked_node_id:
                        edge['data']['source'] += '_clicked'
                    else:
                        edge['data']['source'] += '_connected'
                    print(edge)
                if edge['data']['target'] in connected_node_ids:
                    if edge in connected_edges:
                        edge['style']['line-color'] = node_style
                    if edge['data']['target'] == clicked_node_id:
                        edge['data']['target'] += '_clicked'
                    else:
                        edge['data']['target'] += '_connected'
                    print(edge)

            for node in cluster_nodes:
                if node['data']['id'] in connected_node_ids:
                    node['style'].pop('border-color', None)
                    if node['data']['id'] not in [clicked_node_id,clicked_node_id + '_clicked']:
                        node['data']['id'] += '_connected'
                    node['style']['background-opacity'] = 0.8
                else:
                    node['style']['background-opacity'] = 0.6

                if node['data']['id'] in [clicked_node_id,clicked_node_id + '_clicked']:
                    node['style']['border-opacity'] = 0.7
                    node['data']['id'] += '_clicked'
                    node['style']['border-color'] = 'gold'
                    node['style']['border-width'] = 15
                    node['style']['background-opacity'] = 0.95
                

            if input_id == 'cytoscape-network':
                return cluster_nodes + cluster_edges, content_div, cluster_name
            else:  
                return cluster_nodes + cluster_edges, content_div, dash.no_update
    return dash.no_update, dash.no_update, dash.no_update
# Run app
if __name__ == '__main__':
    app.run_server(debug=True)