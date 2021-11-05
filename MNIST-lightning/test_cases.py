from eval_metrics import eval_metrics


preds = [1,2,3,4,5,6]
targets = [1,2,3,4,5,6]

metrics = eval_metrics(targets,preds)


def test_plt_cm():
    metrics.plot_conf_matx()
    