from bottle import route, run, template, request
from probability_ans import probability_ans


@route('/start')
def start():
    return template('image_template', output_text='')


@route('/start', method='POST')
def start_up():
    # csvs = []
    # for i in xrang(1,5):
    #     csvs.append(request.files.get('input_csv%s'%i))

    input_csv1 = request.files.input_csv1
    input_csv2 = request.files.input_csv2
    input_csv3 = request.files.input_csv3
    input_csv4 = request.files.input_csv4
    input_csv5 = request.files.input_csv5

    file_names = ['usdjpy_d.csv', 'eurpln_d.csv', 'eurusd_d.csv', 'usdjpy_d.csv', 'usdpln_d.csv']


    input_date = request.forms['input_date']
    input_max_depth = request.forms['input_max_depth']
    input_ratio = request.forms['input_ratio']
    output_text = probability_ans(file_names, input_date, input_max_depth, input_ratio)
    output_text += "<br>期間:%s<br>max_depth:%s<br>割合:%s"%(input_date, input_max_depth, input_ratio)
    return template('image_template', output_text=output_text)


run(host='localhost', port=8080, debug=True)
