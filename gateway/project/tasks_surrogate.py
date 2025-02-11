import json
import pickle
import timeit
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.core.files.base import ContentFile
from fpdf.enums import Align, XPos
from smile_ml_core.classes.scoring import MetricName

from file.models import File
from helper.servives.graph_provider import GraphProvider
from helper.servives.structures import NodeModelStructure
from notification.serializers import NotificationSerializer
from notification.views import create_notification
from platformwebserver.celery import app
from platformwebserver.settings import DOMAIN, URL_SCHEME
from polygon.models import Competition, ExtraParamsSurrogate
from polygon.surrogate_modeling_services import call_software, get_software_info
from project.models import Project
from project.services.scores_table import (
    _get_metrics_dict,
    get_best_module_id,
    get_model_node_id,
)
from project.tasks import PdfBlock, PdfBlockType, _generate_pdf
from project.views_graph import get_profile_id
from services.clients.graph.v1 import graph_client
from services.schemas.graph import SavedFilePath
from userprofile.models import UserProfile
from userprofile.services import send_to_user

MEDIA_PATH_TMP = Path('media/tmp') / uuid4().hex
MEDIA_PATH_TMP.mkdir(parents=True, exist_ok=True)

FONT_DEFAULT = ('Inter', '', 12)
FONT_TITLE = ('Inter', 'B', 14)
FONT_SUBTITLE = ('Inter', 'B', 12)
FONT_TABLE = ('Inter', '', 10)

TAB = ' ' * 6


@dataclass
class TableRow2d:
    col1: str
    col2: str
    col3: str = ''
    col4: str = ''
    col5: str = ''
    col6: str = ''
    # col7: str = ''

    def cnt(self):
        values = self.values()
        return len(values)

    def thead(self):
        values = self.values()
        cnt = len(values)
        w1 = 20
        if cnt == 2:
            w1 = 70
        elif cnt == 3:
            w1 = 60
        elif cnt == 4:
            w1 = 50
        elif cnt == 5:
            w1 = 45
        elif cnt == 6:
            w1 = 40
        elif cnt == 7:
            w1 = 35

        wn = int((100 - w1) / (cnt - 1))
        th1 = f'<th width="{w1}%">{values[0]}</th>'
        ths = ''.join(list(map(lambda v: f'<th width="{wn}%">{v}</th>', values[1:])))
        return f'<thead><tr>{th1}{ths}</tr></thead>'

    def tr(self):
        row = '<tr>'
        for v in self.values():
            # if isinstance(v, Grade):
            #     v = v.get()
            row += f'<td>{v}</td>'
        row += '</tr>'

        return row

    def values(self) -> List[str]:
        values = [
            self.col1,
            self.col2,
            self.col3,
            self.col4,
            self.col5,
            self.col6,
            # self.col7,
        ]
        return list(filter(lambda v: v is not None, values))


def _create_metrics_table(user_id: int, module_baseline_ids: list[int]):
    res_scores_table = [TableRow2d(*['Модель', 'SMAPE', 'MAPE', 'MAE', 'MSE', 'R2'])]

    for module_id in module_baseline_ids:
        module = Project.objects.get(id=module_id)

        metrics = _get_metrics_dict(module_id, user_id)
        df_metrics_as_columns = pd.DataFrame.from_records([metrics])
        df_metrics_as_columns.insert(0, 'module_name', module.name)
        columns_to_drop = list(
            map(
                lambda m: m.name.lower(),
                [MetricName.EXPLAINED_VARIANCE, MetricName.MAX_ERROR, MetricName.MEAN_SQUARED_LOG_ERROR],
            )
        )
        df_scores = df_metrics_as_columns.drop(columns=columns_to_drop)

        res_scores_table.extend(df_scores.apply(lambda row: TableRow2d(*row.values.tolist()), axis=1).to_list())

    return res_scores_table


def _get_sampling_points(project_id: int, user_id: int) -> list[pd.DataFrame]:
    sampling_model_name = ExtraParamsSurrogate.objects.get(baseline_module__id=project_id).sampling_method

    graph = GraphProvider(module_id=project_id, user_id=user_id)
    sampling_model_nodes: list[NodeModelStructure] = graph.filter_module(sampling_model_name)

    if len(sampling_model_nodes) != 1:
        raise Exception('Baseline должен иметь один узел сэмплирования')

    sampling_model_node = sampling_model_nodes[0]

    dfs_out = map(pd.DataFrame.from_records, sampling_model_node.data.dfs_out)

    return list(dfs_out)


def _plot_sampling_points(project_id: int, user_id: int) -> BytesIO:
    import itertools
    import math

    dfs_out = _get_sampling_points(project_id, user_id)
    columns = dfs_out[0].columns

    k = 3
    if len(columns) > 1:
        df_all = pd.DataFrame()
        for i, df_out in enumerate(dfs_out):
            df_out['iter'] = i
            df_all = pd.concat([df_all, df_out])

        axs_cnt = math.comb(len(columns), 2)
        ncols_max = 3
        ncols = min(ncols_max, axs_cnt)
        nrows = max(1, axs_cnt // ncols_max)

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * k, nrows * k))

        for ax, (x, y) in zip(axs.flat if axs_cnt > 1 else [axs], itertools.combinations(columns, 2)):
            sc = ax.scatter(x, y, c='iter', data=df_all, cmap='coolwarm')
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(sc, ax=axs, location='bottom', shrink=1 / ncols, label='Итерация')
    else:
        fig, ax = plt.subplots(figsize=(k, k))

        s_init = dfs_out[0].squeeze()
        s_ext = pd.concat(dfs_out[1:]).squeeze()

        ax.hist([s_init, s_ext], bins=10, stacked=True)

        fig.legend(['initial points', 'update points'])  # loc='outside upper center'

    img_buf = BytesIO()
    # fig.tight_layout()
    fig.savefig(img_buf, dpi=500)
    plt.close(fig)

    return img_buf


def _plot_validation_dataset(
    user_id: int,
    profile_id: int,
    node_id: int,
    project_id: int,
    features_limits: dict,
    features: list,
    target_column: str,
    connection_params: dict,
    pdf_blocks: list[PdfBlock],
):
    software_limits = dict(filter(lambda kv: features_limits[kv[0]]['sampling_input'], features_limits.items()))
    target_label = features_limits[target_column]['label']

    mean_limits = {k: [(v['upper'] + v['lower']) / 2] for k, v in software_limits.items()}
    df_mean_limits = pd.DataFrame.from_dict(mean_limits)
    num_val_points = 25

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3))

    for feature, limits in software_limits.items():
        if feature == target_column:
            continue

        points = np.linspace(limits['lower'], limits['upper'], num_val_points)
        df = pd.DataFrame(points, columns=[feature])
        df_means = df_mean_limits.loc[:, df_mean_limits.columns != feature]
        df = df.join(df_means, how='cross')

        start = timeit.default_timer()
        df_test = call_software(df, **connection_params)
        time_software = timeit.default_timer() - start

        df_test = df.join(df_test, how='inner')
        columns = features

        start = timeit.default_timer()

        userprofile = UserProfile.objects.get(id=profile_id)
        project = Project.objects.get(id=project_id)

        file_obj = File.objects.create(
            project_owner=project,
            user_uploaded=userprofile,
            file=ContentFile(pickle.dumps(df_test[columns], protocol=4), 'df_val'),
        )
        file_path = SavedFilePath(file=file_obj.file.url, user_id=user_id, module_id=project_id)
        predict = graph_client.model_nodes.apply_to_new_data(node_id=node_id, payload=file_path)

        file_obj.delete()

        time_model = timeit.default_timer() - start

        df_pred = pd.DataFrame.from_dict(predict)
        df_pred.index = pd.to_numeric(df_pred.index)
        df_test = df_test.join(df_pred)

        df_test.plot.line(x=feature, y=target_column, c='g', ax=ax, label='Оригинальное ПО')
        df_test.plot.line(x=feature, y='predict', c='r', ax=ax, label='Суррогатная модель')

        ax.legend()
        ax.set_xlabel(limits['label'])
        ax.set_ylabel(target_label)

        img_buf = BytesIO()
        fig.tight_layout()
        fig.savefig(img_buf, dpi=500)
        ax.cla()

        pdf_blocks.append(PdfBlock(PdfBlockType.Picture, img_buf))
        text = f'Зависимость "{target_column}" от "{feature}" на валидационном наборе данных'
        pdf_blocks.append(PdfBlock(PdfBlockType.PictureTitle, text))

        text = f'Время работы оригинального программного обеспечения: {time_software:.3f} сек.'
        pdf_blocks.append(PdfBlock(PdfBlockType.Text, text))
        text = f'Время работы построенной суррогатной модели: {time_model:.3f} сек.'
        pdf_blocks.append(PdfBlock(PdfBlockType.Text, text))

    plt.close(fig)


@app.task
def generate_polygon_surrogate_report(user_id: int, competition_id: int):
    """generate_polygon_surrogate_report"""
    profile_id = get_profile_id(user_id)
    competition = Competition.objects.get(id=competition_id)
    module_baseline_ids = list(competition.baseline_module.values_list('id', flat=True))

    pdf_blocks: list[PdfBlock] = []

    connection_params = {
        'url': competition.params_surrogate.url,
        'proxy': competition.params_surrogate.proxy,
        'username': competition.params_surrogate.username,
        'password': competition.params_surrogate.password,
        'timeout': competition.params_surrogate.timeout,
    }

    software_info = get_software_info(**connection_params)

    text = '\n'.join(['Программное обеспечение: {name}.', 'Версия: {version}.', 'Описание задачи: {description}.'])
    pdf_blocks.append(PdfBlock(PdfBlockType.Summary, text.format(**software_info)))

    competition_name = competition.name
    target_column = competition.target_column

    text = (
        f'Ниже приведены результаты применения шаблона "{competition_name}" для суррогатной модели на базе методов '
        'сэмплирования библиотеки Surrogate Modeling Toolbox и регрессионных моделей. Целевым признаком '
        f'прогнозирования является признак "{target_column}". Шаблон "{competition_name}" направлен на построение '
        'суррогатной модели индустриального программного обеспечения. Проверка качества осуществляется по метрикам '
        'качества регрессионных моделей (оценивание вероятностей получения верных предсказаний). Ниже приведена '
        'таблица с оценками качества суррогатных моделей.'
    )
    pdf_blocks.append(PdfBlock(PdfBlockType.Text, text))

    text = 'Раздел 1. Сравнение метрик качества различных моделей, построенных в ПК ПОЛИСУРМ-Р'
    pdf_blocks.append(PdfBlock(PdfBlockType.Title, text))

    target_metric = MetricName(competition.metric).name.lower()

    metrics_table = _create_metrics_table(user_id=user_id, module_baseline_ids=module_baseline_ids)
    best_project_id, best_target_metric_value = get_best_module_id(
        module_baseline_ids, target_metric, user_id=profile_id
    )

    pdf_blocks.append(PdfBlock(PdfBlockType.TableTitle, 'Метрики качества суррогатных моделей'))
    pdf_blocks.append(PdfBlock(PdfBlockType.Table, metrics_table))

    best_project = Project.objects.get(id=best_project_id)

    model_name = (
        'FedotModel'
        if best_project.extra_params_surrogate.include_auto_ml
        else best_project.extra_params_surrogate.cv_model
    )
    node_id = get_model_node_id(best_project_id, user_id, model_name)

    text = f'Рекомендуемой суррогатной моделью является "{best_project.name}".'
    if len(module_baseline_ids) > 1:
        text = 'Пользователь задал в испытании сравнение нескольких шаблонов суррогатного моделирования. ' + text
    pdf_blocks.append(PdfBlock(PdfBlockType.Text, text))

    text = 'Раздел 2. Результаты исследования рекомендуемой суррогатной модели'
    pdf_blocks.append(PdfBlock(PdfBlockType.Title, text))

    extra_params = ExtraParamsSurrogate.objects.get(baseline_module=best_project)

    text = (
        'Ниже представлено графическое сравнение исходного пространства точек (initial points), полученных с помощью '
        f'метода сэмплирования {extra_params.sampling_method} и итогового пространства точек после итерационного '
        f'расширения плана эксперимента (update points) для рекомендуемой суррогатной модели "{best_project.name}". '
        'В итоговом плане эксперимента для большого количества сэмплов выделяются кластеры в местах с наибольшей '
        'дисперсией.'
    )
    pdf_blocks.append(PdfBlock(PdfBlockType.Text, text))

    img = _plot_sampling_points(best_project_id, user_id)
    pdf_blocks.append(PdfBlock(PdfBlockType.Picture, img))

    text = 'Диаграмма рассеяния исходного и итогового распределения значений входных переменных'
    pdf_blocks.append(PdfBlock(PdfBlockType.PictureTitle, text))

    features_limits = json.loads(competition.params_surrogate.features_limits)
    features = json.loads(competition.features)

    _plot_validation_dataset(
        user_id,
        profile_id,
        node_id,
        best_project_id,
        features_limits,
        features,
        target_column,
        connection_params,
        pdf_blocks,
    )

    text = (
        'Раздел 3. Итоговое заключение о практических результатах и рекомендациях по использованию рекомендуемой '
        'модели, построенной в ПК ПОЛИСУРМ-Р'
    )
    pdf_blocks.append(PdfBlock(PdfBlockType.Title, text))

    c = -1 if MetricName(competition.metric) == MetricName.R2 else 1
    if c * (extra_params.metric_threshold - best_target_metric_value) >= 0:
        metric_result = 'Модель достигла требуемого порога точности.'
    else:
        metric_result = (
            'Модель не достигла требуемого порога точности. Рекомендуется провести дополнительные эксперименты.'
        )

    summary_lines = [
        f'Целевой метрикой является "{target_metric}", достигнутое значение метрики: {best_target_metric_value}.',
        f'Заданный порог метрики: {extra_params.metric_threshold}',
        f'{metric_result}',
        f'Целевым признаком прогнозирования является признак "{target_column}".',
        'Для создания суррогатного аналога индустриального программного обеспечения рекомендуется использовать модель '
        f'"{best_project.name}", обученную с параметрами:',
        f'Исходное количество точек сэмплирования: {extra_params.count_gen_sampling_points}',
        f'Количество точек приращения: {extra_params.count_inc_sampling_points}',
        f'Размер окна доопределения данных: {extra_params.window_size}',
        f'Количество итераций: {extra_params.count_iter}',
        'Границы значений входных параметров:',
    ]

    for feature, limits in features_limits.items():
        if limits['sampling_input']:
            summary_lines.append(f'{limits["label"]} от {limits["lower"]} до {limits["upper"]}')

    for summary_line in summary_lines:
        pdf_blocks.append(PdfBlock(PdfBlockType.Summary, summary_line))

    title = 'Отчет по проведению эксперимента суррогатного моделирования'
    pdf = _generate_pdf(title)
    fig_i, table_i = 1, 1

    for pdf_block in pdf_blocks:
        if pdf_block.block_type == PdfBlockType.Text:
            pdf.multi_cell(txt=pdf_block.block, w=0, new_x=XPos.LMARGIN)
        elif pdf_block.block_type == PdfBlockType.Title:
            pdf.set_font(*FONT_SUBTITLE)
            pdf.ln(5)
            pdf.multi_cell(txt=pdf_block.block, w=0, new_x=XPos.LMARGIN)
            pdf.ln(5)
            pdf.set_font(*FONT_DEFAULT)
        elif pdf_block.block_type == PdfBlockType.Summary:
            pdf.multi_cell(txt=pdf_block.block, w=0, new_x=XPos.LMARGIN)
            pdf.ln(2)
        elif pdf_block.block_type == PdfBlockType.Picture:
            pdf.image(pdf_block.block, h=75, x=Align.C)
        elif pdf_block.block_type == PdfBlockType.HTML:
            pdf.write_html(pdf_block.block)
        elif pdf_block.block_type == PdfBlockType.Table:
            block0: TableRow2d = pdf_block.block[0]
            thead = block0.thead()

            tbody = '<tbody>'
            for block in pdf_block.block[1:]:  # type: TableRow2d
                tbody += block.tr()
            tbody += '</tbody>'
            table = f"""
                <table width="95%">
                {thead}
                {tbody}
                </table>
            """
            pdf.set_font(*FONT_TABLE)
            pdf.write_html(table)
            pdf.set_font(*FONT_DEFAULT)
        elif pdf_block.block_type == PdfBlockType.TableTitle:
            pdf.multi_cell(txt=f'Таблица {table_i} - {pdf_block.block}\n', w=0, new_x=XPos.LMARGIN)
            table_i += 1
        elif pdf_block.block_type == PdfBlockType.PictureTitle:
            pdf.multi_cell(txt=f'Рисунок {fig_i} - {pdf_block.block}\n', w=0, new_x=XPos.LMARGIN, align=Align.C)
            fig_i += 1

    pdf_filename = MEDIA_PATH_TMP / 'final_report.pdf'
    pdf.output(str(pdf_filename))

    # link = (Path.cwd() / pdf_filename).as_posix()
    link = URL_SCHEME + DOMAIN + '/api/' + pdf_filename
    message = f'<p>Скачать отчет для испытания {competition_name} можно по&nbsp;<a href="{link}" target="_blank">ссылке</a></p>'
    notifications = create_notification(
        profile_to=profile_id, title=f'Отчет для испытания {competition_name} был сгенерирован', message=message
    )

    for notification in notifications:
        send_to_user('update.notifications', data=NotificationSerializer(notification).data, user_id=user_id)
