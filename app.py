import dash
from dash import dcc, html, Input, Output, State, callback, ctx, clientside_callback, no_update
import dash_player as dp
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "NRL Match Set Video Reviewer"
server = app.server


video_dict = {
    1:
        {
            0: 'https://nrl.rugby.league.prozone.s3.ap-southeast-2.amazonaws.com/NRL_lvl1_Statistics/NRL_Premiership/2025/Video/NRL_252804_CAN_BRI_CAM1_H1.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA5V5NPAF7JCAR4TGY/20250916/ap-southeast-2/s3/aws4_request&X-Amz-Date=20250916T044324Z&X-Amz-Expires=604791&X-Amz-Signature=ad087bdc42e9634077db2cb12107e9e24b451aa2c5f08504999368661d024acc&X-Amz-SignedHeaders=host&response-content-disposition=inline',
            1: 'https://nrl.rugby.league.prozone.s3.ap-southeast-2.amazonaws.com/NRL_lvl1_Statistics/NRL_Premiership/2025/Video/NRL_252804_CAN_BRI_PGM_H1.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA5V5NPAF7JCAR4TGY/20250916/ap-southeast-2/s3/aws4_request&X-Amz-Date=20250916T044428Z&X-Amz-Expires=604795&X-Amz-Signature=92d2b80a024b82d1d2d6b52591743e1d752c3d63288de7395e794aaef06918a5&X-Amz-SignedHeaders=host&response-content-disposition=inline',
        },
    2:
        {
            0: 'https://nrl.rugby.league.prozone.s3.ap-southeast-2.amazonaws.com/NRL_lvl1_Statistics/NRL_Premiership/2025/Video/NRL_252804_CAN_BRI_CAM1_H2.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA5V5NPAF7JCAR4TGY/20250916/ap-southeast-2/s3/aws4_request&X-Amz-Date=20250916T044456Z&X-Amz-Expires=604797&X-Amz-Signature=79151512fb99a1b7fddd8624db390f77fd757f8c99a728fd8f619ac5e819b9c2&X-Amz-SignedHeaders=host&response-content-disposition=inline',
            1: 'https://nrl.rugby.league.prozone.s3.ap-southeast-2.amazonaws.com/NRL_lvl1_Statistics/NRL_Premiership/2025/Video/NRL_252804_CAN_BRI_PGM_H2.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA5V5NPAF7JCAR4TGY/20250916/ap-southeast-2/s3/aws4_request&X-Amz-Date=20250916T044525Z&X-Amz-Expires=604795&X-Amz-Signature=ce8b9c09acd49a59624bff7645a1667fb909bef8f064964525d65699e7a09b5f&X-Amz-SignedHeaders=host&response-content-disposition=inline'
        }
}

#what to subtract off each vr for each half to sync to the right time
offset_dict = {
    1: 0,
    2: -78754,
    3: -78754,
    4: -78754,
    5: -78754,
}

play_by_play_df = pd.read_excel("data/play_by_play_20251112840.xlsx")
play_by_play_df['vr'] = play_by_play_df['min'] + play_by_play_df['Half'].map(offset_dict)
play_by_play_df['vr_end'] = play_by_play_df['max'] + play_by_play_df['Half'].map(offset_dict)

team_dims_df = pd.read_csv("data/team_dims.csv")

def generate_set_df(df):
    incompletions = ['Error','Kick Dead','Kick Caught In Goal']
    index_cols = ['Half','SC','AttackingTeam','DefendingTeam']
    start_cols = ['min','vr','TN','StartType','StartX']
    finish_cols = ['TN','FinishType','FinishX','max','vr_end']

    start_df = (df.query("TN != 99 and SetStart == 1").set_index(index_cols)[start_cols]
                .rename(columns={"TN":"StartTN"}))
    finish_df = (df.query("TN != 99 and SetFinish == 1").set_index(index_cols)[finish_cols]
                 .rename(columns={"TN":"FinishTN"}))
    set_df = pd.concat([start_df,finish_df],axis=1)

    #Completions
    set_df['Completion'] = 1
    set_df.loc[set_df['FinishType'].isin(incompletions),'Completion'] = 0
    #penalty completions
    condition = (
            (df["FinishType"] == "Penalty") &
            (df["StartType"].shift(-1) == "Penalty Tap") &
            (df["AttackingTeam"] != df["AttackingTeam"].shift(-1))
    )
    df.loc[condition, "Completion"] = 0

    set_df['SetDistance'] = set_df['FinishX'] - set_df['StartX']
    set_df['Plays'] = set_df['FinishTN'] - set_df['StartTN'] + 1
    set_df["Text1"] = (set_df["SetDistance"] / 10).round().astype(int).astype(str) + "m, " + set_df["Plays"].astype(
        str) + " plays, " + set_df["FinishType"]
    set_df["Text"] = set_df["FinishType"]

    return set_df.reset_index()

def generate_set_chart(df,team_dim):
    teams = np.sort(df.AttackingTeam.unique())
    team = teams[0]
    opp = teams[1]
    team_col = team_dim.set_index('teamname').loc[team,'teamhexcolour']
    opp_col = team_dim.set_index('teamname').loc[opp,'teamhexcolour']

    incomplete_col = 'red'
    complete_col = 'green'

    fig = go.Figure()

    #Add direction triangles
    #team complete
    fig.add_trace(go.Scatter(
        x=df.query("AttackingTeam == @team and Completion == 1").FinishX + 10,
        y=df.query("AttackingTeam == @team and Completion == 1").SC,
        mode='markers',
        opacity=1,
        marker=dict(
            symbol='triangle-right',
            size=20,
            color=complete_col,
            line=dict(
                width=0,
                color="black"
            )
        )
    ))
    #team incomplete
    fig.add_trace(go.Scatter(
        x=df.query("AttackingTeam == @team and Completion == 0").FinishX + 10,
        y=df.query("AttackingTeam == @team and Completion == 0").SC,
        mode='markers',
        opacity=1,
        marker=dict(
            symbol='triangle-right',
            size=20,
            color=incomplete_col,
            line=dict(
                width=0,
                color="black"
            )
        )
    ))
    #opp complete
    fig.add_trace(go.Scatter(
        x=1000 - df.query("AttackingTeam != @team and Completion == 1").FinishX - 10,
        y=df.query("AttackingTeam != @team and Completion == 1").SC,
        mode='markers',
        marker=dict(
            symbol='triangle-left',
            size=20,
            color=complete_col,
            line=dict(
                width=0,
                color=opp_col
            )
        )
    ))
    #opp incomplete
    fig.add_trace(go.Scatter(
        x=1000 - df.query("AttackingTeam != @team and Completion == 0").FinishX - 10,
        y=df.query("AttackingTeam != @team and Completion == 0").SC,
        mode='markers',
        marker=dict(
            symbol='triangle-left',
            size=20,
            color=incomplete_col,
            line=dict(
                width=0,
                color=opp_col
            )
        )
    ))
    #Add Bars
    #team gain
    fig.add_trace(go.Bar(
        x=df.query("AttackingTeam == @team and SetDistance >= 0").SetDistance,
        base=df.query("AttackingTeam == @team and SetDistance >= 0").StartX,
        y=df.query("AttackingTeam == @team and SetDistance >= 0").SC,
        orientation='h',
        marker=dict(
            color=team_col,
            line=dict(
                color=team_col,
                width=2
            )
        ),
        width=0.7,
        text='   ' + df.query("AttackingTeam == @team and SetDistance >= 0").Text,
        textposition='outside',
        customdata=df.query("AttackingTeam == @team and SetDistance >= 0")[['Half','vr']].values
    ))
    #team loss
    fig.add_trace(go.Bar(
        x=df.query("AttackingTeam == @team and SetDistance < 0").SetDistance * -1,
        base=df.query("AttackingTeam == @team and SetDistance < 0").FinishX,
        y=df.query("AttackingTeam == @team and SetDistance < 0").SC,
        orientation='h',
        marker=dict(
            color='rgba(0, 0, 0, 0)',
            line=dict(
                color=team_col,
                width=2
            )
        ),
        width=0.7,
        text=df.query("AttackingTeam == @team and SetDistance < 0").Text,
        textposition='outside',
        customdata=df.query("AttackingTeam == @team and SetDistance < 0")[['Half','vr']].values
    ))
    #opp gain
    fig.add_trace(go.Bar(
        x=df.query("AttackingTeam != @team and SetDistance >= 0").SetDistance * -1,
        base=1000 - df.query("AttackingTeam != @team and SetDistance >= 0").StartX,
        y=df.query("AttackingTeam != @team and SetDistance >= 0").SC,
        orientation='h',
        marker=dict(
            color=opp_col,
            line=dict(
                color=opp_col,
                width=2
            )
        ),
        width=0.7,
        text=df.query("AttackingTeam != @team and SetDistance >= 0").Text + '   ',
        textposition='outside',
        customdata=df.query("AttackingTeam != @team and SetDistance >= 0")[['Half','vr']].values
    ))
    #opp loss
    fig.add_trace(go.Bar(
        x=df.query("AttackingTeam != @team and SetDistance < 0").SetDistance,
        base=1000 - df.query("AttackingTeam != @team and SetDistance < 0").FinishX,
        y=df.query("AttackingTeam != @team and SetDistance < 0").SC,
        orientation='h',
        marker=dict(
            color='rgba(0, 0, 0, 0)',
            line=dict(
                color=opp_col,
                width=2
            )
        ),
        width=0.7,
        text=df.query("AttackingTeam != @team and SetDistance < 0").Text,
        textposition='outside',
        customdata=df.query("AttackingTeam != @team and SetDistance < 0")[['Half','vr']].values
    ))

    #Add start type annotation
    annotation_style = dict(
        font=dict(
            family='Arial Black',
            size=10
        ),
    )
    # team gain
    for idx, row in df.query("AttackingTeam == @team and SetDistance >= 0").iterrows():
        fig.add_annotation(
            x=row["StartX"],
            y=row["SC"],
            xref="x",
            yref="y",
            text=row["StartType"],
            showarrow=False,
            xanchor='right',
            # **annotation_style
        )
    #team loss
    for idx, row in df.query("AttackingTeam == @team and SetDistance < 0").iterrows():
        fig.add_annotation(
            x=row["FinishX"],
            y=row["SC"],
            xref="x",
            yref="y",
            text=row["StartType"],
            showarrow=False,
            xanchor='right',
            # **annotation_style
        )
    # opposition gain
    for idx, row in df.query("AttackingTeam != @team and SetDistance >= 0").iterrows():
        fig.add_annotation(
            x=1000 - row["StartX"],
            y=row["SC"],
            xref="x",
            yref="y",
            text=row["StartType"],
            showarrow=False,
            xanchor='left',
            # **annotation_style
        )
    # opposition loss
    for idx, row in df.query("AttackingTeam != @team and SetDistance < 0").iterrows():
        fig.add_annotation(
            x=1000 - row["FinishX"],
            y=row["SC"],
            xref="x",
            yref="y",
            text=row["StartType"],
            showarrow=False,
            xanchor='left',
            # **annotation_style
        )

    fig.update_yaxes(range=[df.SC.max() + 0.5, df.SC.min() - 0.5], dtick=1,fixedrange=True)
    fig.update_xaxes(tickvals=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                     ticktext=["TL", "10m", "20m", "30m", "40m", "50m", "40m", "30m", "20m", "10m", "TL"],
                     ticklabelposition="outside",
                     ticks="inside",
                     side="top",
                     mirror='allticks',
                     range=[-150, 1150],fixedrange=True,)

    #Add field markings
    fig.add_vline(x=-100, line_width=2, line_color="black")
    fig.add_vline(x=1100, line_width=2, line_color="black")
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.add_vline(x=1000, line_width=1, line_color="black")
    fig.add_vline(x=200, line_width=1, line_color="red")
    fig.add_vline(x=800, line_width=1, line_color="red")
    fig.add_vline(x=500, line_width=1, line_color="black", line_dash="dash")

    #add half lines
    for half,sc in df.groupby('Half')['SC'].first().items():
        if half > 1:
            fig.add_hline(y=sc-0.5, line_width=6, line_color="grey", annotation_text=f"Half {half}",annotation_position='bottom right')

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",   # plot area transparent
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, b=10, t=10,pad=10),
        autosize=True,
        dragmode=False,
    )
    return fig

set_df = generate_set_df(play_by_play_df)
set_fig = generate_set_chart(set_df,team_dims_df)
set_df_indexed = set_df.set_index("SC")

button_group = dbc.ButtonGroup(
    [
        dbc.RadioItems(
            id="button-angles",
            options=[
                {"label": "Broadcast", "value": 0},
                {"label": "Eagle", "value": 1},
            ],
            value=0,
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
        ),
    ],
    size="md",  # optional
)

app.layout = html.Div(
    dbc.Container(
        [
            # dcc.Interval(id="interval", interval=2000),
            # dcc.Store(id="store-video-time"),
            # dcc.Store(id="store-video-half"),
            dbc.Row(
                dbc.Col(
                    button_group,
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dp.DashPlayer(id="video-player",controls=True,playing=True,intervalCurrentTime=1000,
                                          url=video_dict[1][0],
                                          width="100%",
                                          height="100%"
                                          ),
                        ],width=8
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(id="chart",
                                      figure=set_fig,
                                      config={"displayModeBar": False},
                                      style={'width': '100%', 'height': '3000px'}),
                        ],width=4, style={"height": "100%","overflowY":"scroll"}
                    ),
                ],
                style={"height": "100%"}
            ),
        ],
        fluid=True,
        style={"height": "100%","padding":"10px"},
    ),className="pagenoscroll"
)


@app.callback(
Output("video-player","seekTo"),
    Output("video-player", "url"),
    Input("chart", "clickData"),
    State("button-angles", "value"),
    State("video-player","url"), prevent_initial_call=True
)
def update_video_time(clickData,angle,url):

    if clickData is None:
        return f'{video_dict[1][angle]}',0,0

    sc = clickData["points"][0]["y"]
    half = set_df_indexed.loc[sc,'Half']
    vr = set_df_indexed.loc[sc,'vr']
    time = round(vr/25) - 3
    if half > 2:
        half = 2
    new_url = f'{video_dict[half][angle]}'

    if new_url == url:
        return time, no_update

    return time, new_url




# # Clientside to capture the video time
# clientside_callback(
#     """
#     function(n_intervals) {
#         if (n_intervals === 0) return 0;  // Skip first call
#
#         const video = document.getElementById('video');
#         if (video && video.readyState >= 2) {  // Wait until video metadata is loaded
#             return video.currentTime;
#         }
#         return 0;
#     }
#     """,
#     Output("store-video-time", "data",allow_duplicate=True),
#     Input("interval", "n_intervals"), prevent_initial_call=True,
# )


#Works
@app.callback(
    Output("video-player", "url",allow_duplicate=True),
    Output("video-player", "seekTo",allow_duplicate=True),
    Input("button-angles", "value"),
    State("video-player", "url"),
    State("video-player", "currentTime"),prevent_initial_call=True,
)
def switch_angle(angle,url,time):
    # get video half
    half = int(url.split('?')[0][-5])
    #set new url
    new_url = f'{video_dict[half][angle]}'

    #if not time start at 0
    if time is None:
        return new_url, 0
        
    return new_url,round(time)-1

#Works
@app.callback(
    Output("chart", "figure", allow_duplicate=True),
    Input("video-player", "currentTime"),
    State("chart", "figure"),
    State("video-player","url"),
    prevent_initial_call=True,
)
def update_chart(time,fig_json, url):
    fig = go.Figure(fig_json)

    #get video half
    half = int(url.split('?')[0][-5])
    if time is None:
        vr = 1
    else:
        vr = (time+3)*25

    #get first index based on time
    if half == 1:
        sc = set_df_indexed.loc[(set_df_indexed['Half']==half) & (set_df_indexed['vr_end'] >= vr)]
    else:
        sc = set_df_indexed.loc[(set_df_indexed['Half'] >= half) & (set_df_indexed['vr_end'] >= vr)]


    # Keep everything except rectangles
    static_shapes = [s for s in fig.layout.shapes if s["type"] != "rect"]
    # Replace layout.shapes with only the non-rectangles
    fig.layout.shapes = tuple(static_shapes)

    if len(sc) == 0:
        return fig

    sc = sc.index[0]

    fig.add_hrect(y0=sc-0.5,y1=sc+0.5, line_width=0, fillcolor="grey",opacity=0.3)

    return fig


if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=8080,use_reloader=True)
