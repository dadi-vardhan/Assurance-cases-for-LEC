import sys
sys.path.insert(0, '/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/MNIST-lightning')
from eval_metrics import eval_metrics


preds = [0,1,2,3,4,5,6,7,8,9]
targets = [0,1,2,3,4,5,6,7,8,9]
classes = ('Zero', 'One', 'Two', 'Three', 'Four',
                            'Five', 'Six', 'Seven', 'Eight', 'Nine')

metrics = eval_metrics(targets,preds,classes)



def test_plot_confusion_matrix():
    
    try:
        fig = metrics.plot_conf_matx()
    except:
        assert False
    else:
        assert True

def test_classification_report():
    
    try:
        metrics.classify_report()
    except:
        assert False
    else:
        assert True

def test_accuracy():
    
    try:
        metrics.accuracy()
    except:
        assert False
    else:
        assert True
        
def test_precision():
    
    try:
        metrics.precision_weighted()
    except:
        assert False
    else:
        assert True
        
def test_recall():
    
    try:
        metrics.recall_weighted()
    except:
        assert False
    else:    
        assert True
        
def test_f1_score():
    
    try:
        metrics.f1_score_weighted()
    except:
        assert False
    else:
        assert True
    
if __name__ == "__main__":
    test_plot_confusion_matrix()
    test_classification_report()
    test_accuracy()
    test_f1_score()
    test_precision()
    test_recall()
    
    