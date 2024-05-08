import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QComboBox, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QGroupBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5 import QtGui


config = {
    "font.family":'Arial',
    "font.size": 15,
    # "mathtext.fontset":'stix',
}
rcParams.update(config)

def section_type_to_hot_code(section_type):
    if section_type == 'Rectangular':
        #               B  F  R
        section_list = [0, 0, 1]
    if section_type == 'Barbell':
        #               B  F  R
        section_list = [1, 0, 0]
    if section_type == 'Flange':
        #               B  F  R
        section_list = [0, 1, 0]
    return section_list

def normalize(original_data, mean, var, min_data, max_data):
    normalized_standard = np.true_divide((original_data-mean), np.sqrt(var))
    normalized_min_max  = np.true_divide((normalized_standard-min_data), (max_data-min_data))
    return normalized_min_max

def back_from_normalized(normalized_data, mean, var, min_data, max_data):
    back_from_min_max = normalized_data*(max_data-min_data)+min_data
    back_from_standard = back_from_min_max*np.sqrt(var)+mean
    return back_from_standard

def predictor_fm(capacity_ratio, shear_span, axial_ratio, longi_reinf, hoop_reinf,
                width_to_thick, web_hor_reinf, web_ver_reinf, Ab_Ag, section_type):
    
    features_list = [shear_span, width_to_thick, web_ver_reinf, web_hor_reinf, longi_reinf, hoop_reinf,
                    axial_ratio, Ab_Ag, capacity_ratio]
    section_list  = section_type_to_hot_code(section_type)
    features_list = features_list + section_list
    features_np   = np.array(features_list)

    with open(sys.path[0]+'/Scaler_fm.txt') as scaler_fm:
        lines = scaler_fm.readlines()
        fm_mean = lines[3].split()
        fm_var  = lines[5].split()
        fm_min  = lines[8].split()
        fm_max  = lines[10].split()
        fm_mean_np = np.array(fm_mean, dtype=float)
        fm_var_np  = np.array(fm_var, dtype=float)
        fm_min_np  = np.array(fm_min, dtype=float)
        fm_max_np  = np.array(fm_max, dtype=float)

    features_normalized = normalize(original_data=features_np, mean=fm_mean_np, var=fm_var_np,
                                    min_data=fm_min_np, max_data=fm_max_np)

    fm_predictor = joblib.load(sys.path[0]+'/fm_xgboost.pkl')
    fm_predicted = fm_predictor.predict(features_normalized.reshape(1,-1))
    if fm_predicted[0] == '1' or fm_predicted[0] == 1:
        fm_name = 'Flexure'
    if fm_predicted[0] == '2' or fm_predicted[0] == 2:
        fm_name = 'Flexure-Shear'
    if fm_predicted[0] == '3' or fm_predicted[0] == 3:
        fm_name = 'Shear'
    if fm_predicted[0] == '4' or fm_predicted[0] == 4:
        fm_name = 'Sliding'
    
    return fm_name

def predictor_strength(shear_span, axial_ratio, longi_reinf, hoop_reinf,
                width_to_thick, web_hor_reinf, web_ver_reinf, Ab_Ag, section_type):
    
    features_list = [shear_span, width_to_thick, web_ver_reinf, web_hor_reinf, longi_reinf, hoop_reinf,
                    axial_ratio, Ab_Ag]
    section_list  = section_type_to_hot_code(section_type)
    features_list = features_list + section_list
    features_np   = np.array(features_list)

    with open(sys.path[0]+'/Scaler_strength.txt') as scaler_strength:
        lines = scaler_strength.readlines()
        strength_mean_x = lines[4].split()
        strength_var_x  = lines[6].split()
        strength_mean_y = lines[9].split()
        strength_var_y  = lines[11].split()
        strength_min_x  = lines[15].split()
        strength_max_x  = lines[17].split()
        strength_min_y  = lines[20].split()
        strength_max_y  = lines[22].split()
        strength_mean_x_np = np.array(strength_mean_x, dtype=float)
        strength_var_x_np  = np.array(strength_var_x, dtype=float)
        strength_mean_y_np = np.array(strength_mean_y, dtype=float)
        strength_var_y_np  = np.array(strength_var_y, dtype=float)
        strength_min_x_np  = np.array(strength_min_x, dtype=float)
        strength_max_x_np  = np.array(strength_max_x, dtype=float)
        strength_min_y_np  = np.array(strength_min_y, dtype=float)
        strength_max_y_np  = np.array(strength_max_y, dtype=float)
        
    features_normalized = normalize(original_data=features_np, mean=strength_mean_x_np,
                                    var=strength_var_x_np, min_data=strength_min_x_np,
                                    max_data=strength_max_x_np)

    strength_predictor = joblib.load(sys.path[0]+'/strength_gb.pkl')
    strength_predicted = strength_predictor.predict(features_normalized.reshape(1,-1))
    
    strength = back_from_normalized(normalized_data=strength_predicted, mean=strength_mean_y_np,
                        var=strength_var_y_np, min_data=strength_min_y_np, max_data=strength_max_y_np)

    return strength[0]

def predictor_deformation(shear_span, axial_ratio, longi_reinf, hoop_reinf,
                width_to_thick, web_hor_reinf, web_ver_reinf, Ab_Ag, section_type):
    
    features_list = [shear_span, width_to_thick, web_ver_reinf, web_hor_reinf, longi_reinf, hoop_reinf,
                    axial_ratio, Ab_Ag]
    section_list  = section_type_to_hot_code(section_type)
    features_list = features_list + section_list
    features_np   = np.array(features_list)

    with open(sys.path[0]+'/Scaler_deformation.txt') as scaler_deformation:
        lines = scaler_deformation.readlines()
        deformation_mean_x = lines[4].split()
        deformation_var_x  = lines[6].split()
        deformation_mean_y = lines[9].split()
        deformation_var_y  = lines[11].split()
        deformation_min_x  = lines[15].split()
        deformation_max_x  = lines[17].split()
        deformation_min_y  = lines[20].split()
        deformation_max_y  = lines[22].split()
        deformation_mean_x_np = np.array(deformation_mean_x, dtype=float)
        deformation_var_x_np  = np.array(deformation_var_x, dtype=float)
        deformation_mean_y_np = np.array(deformation_mean_y, dtype=float)
        deformation_var_y_np  = np.array(deformation_var_y, dtype=float)
        deformation_min_x_np  = np.array(deformation_min_x, dtype=float)
        deformation_max_x_np  = np.array(deformation_max_x, dtype=float)
        deformation_min_y_np  = np.array(deformation_min_y, dtype=float)
        deformation_max_y_np  = np.array(deformation_max_y, dtype=float)
        
    features_normalized = normalize(original_data=features_np, mean=deformation_mean_x_np,
                                    var=deformation_var_x_np, min_data=deformation_min_x_np,
                                    max_data=deformation_max_x_np)

    deformation_predictor = joblib.load(sys.path[0]+'/deformation_rf.pkl')
    deformation_predicted = deformation_predictor.predict(features_normalized.reshape(1,-1))
    
    deformation = back_from_normalized(normalized_data=deformation_predicted, mean=deformation_mean_y_np,
                        var=deformation_var_y_np, min_data=deformation_min_y_np, max_data=deformation_max_y_np)

    return deformation[0]

def plot_rec(origin, height, width, fig_num):
    plt.figure(fig_num)
    # left -> back -> right -> front
    plt.plot([origin[0], origin[0]], [origin[1], origin[1]+height], '-k')
    plt.plot([origin[0], origin[0]+width], [origin[1]+height, origin[1]+height], '-k')
    plt.plot([origin[0]+width, origin[0]+width], [origin[1]+height, origin[1]], '-k')
    plt.plot([origin[0]+width, origin[0]], [origin[1], origin[1]], '-k')
    plt.axis('scaled')
    plt.xlim(-0.2,1.2)
    return None

def plot_rectangular(origin, height, width, thickness, fig_num=1):
    plt.figure(fig_num)
    # elevation
    plot_rec(origin=origin, width=width, height=height, fig_num=fig_num)
    # plan (top view)
    spacing_of_two_view = 0.5*width
    plot_rec(origin=[origin[0],height+spacing_of_two_view], width=width, height=thickness, fig_num=fig_num)
    plt.axis('scaled')
    plt.xlim(-0.2,1.2)
    return None

def plot_barbell(origin, height, width, thickness, ratio_width=0.2, ratio_thickness=1.5, fig_num=1):
    '''
    ratio_width     = 0.2   # ratio of barbell width to wall width
    ratio_thickness = 1.5   # ratio of barbell thickness to wall thickness
    '''
    plt.figure(fig_num)
    origin_x = origin[0]
    origin_y = origin[1]
    # elevation
    barbell_width   = width*ratio_width
    # outline -> two vertical lines inside
    plot_rec(origin=origin, width=width, height=height, fig_num=fig_num)
    plt.plot([origin_x+ratio_width*width, origin_x+ratio_width*width], [origin_y, origin_y+height], '-k')
    plt.plot([origin_x+width-barbell_width, origin_x+width-barbell_width], [origin_y, origin_y+height], '-k')
    # plan (top view)
    spacing_of_two_view = 0.5*width
    barbell_thickness   = thickness*ratio_thickness
    half_thickness_difference = (barbell_thickness-thickness)/2
    # left -> back -> right -> front
    origin_plan_y = origin_y+height+spacing_of_two_view
    plt.plot([origin_x, origin_x], [origin_plan_y, origin_plan_y+barbell_thickness], '-k')
    plt.plot([origin_x, origin_x+barbell_width], [origin_plan_y+barbell_thickness, origin_plan_y+barbell_thickness], '-k')
    plt.plot([origin_x+barbell_width, origin_x+barbell_width],
            [origin_plan_y+barbell_thickness, origin_plan_y+barbell_thickness-(ratio_thickness*thickness-thickness)/2], '-k')
    plt.plot([origin_x+barbell_width, origin_x+width-barbell_width],
            [origin_plan_y+barbell_thickness-half_thickness_difference, origin_plan_y+barbell_thickness-half_thickness_difference], '-k')
    plt.plot([origin_x+width-barbell_width, origin_x+width-barbell_width],
            [origin_plan_y+barbell_thickness-half_thickness_difference, origin_plan_y+barbell_thickness], '-k')
    plt.plot([origin_x+width-barbell_width, origin_x+width], [origin_plan_y+barbell_thickness, origin_plan_y+barbell_thickness], '-k')
    plt.plot([origin_x+width, origin_x+width], [origin_plan_y+barbell_thickness, origin_plan_y], '-k')
    plt.plot([origin_x+width, origin_x+width-barbell_width], [origin_plan_y, origin_plan_y], '-k')
    plt.plot([origin_x+width-barbell_width, origin_x+width-barbell_width],
            [origin_plan_y, origin_plan_y+half_thickness_difference], '-k')
    plt.plot([origin_x+width-barbell_width, origin_x+barbell_width],
            [origin_plan_y+half_thickness_difference, origin_plan_y+half_thickness_difference], '-k')
    plt.plot([origin_x+barbell_width, origin_x+barbell_width], [origin_plan_y+half_thickness_difference, origin_plan_y], '-k')
    plt.plot([origin_x+barbell_width, origin_x], [origin_plan_y, origin_plan_y], '-k')
    plt.axis('scaled')
    plt.xlim(-0.2,1.2)
    
    return None

def plot_wall(section_type, height, width, thickness, fig_num=1):
    plt.figure(fig_num)
    if section_type == 'Rectangular':
        plot_rectangular(origin=[0,0], height=height, width=width, thickness=thickness)
    if section_type == 'Barbell':
        plot_barbell(origin=[0,0], height=height, width=width, thickness=thickness, ratio_width=0.2, ratio_thickness=3)
    if section_type == 'Flange':
        plot_barbell(origin=[0,0], height=height, width=width, thickness=thickness, ratio_width=0.07, ratio_thickness=8)
    plt.axis('scaled')
    plt.xlim(-0.2,1.2)
    # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    # plt.axis('off')
    return None

def read_crack_data(crack_data, line_num):
    with open(sys.path[0]+crack_data) as crack:
        lines = crack.readlines()
        line_name_list = ['Line #{0}\n'.format(i) for i in range(1, line_num+1)]
        lines_row_num = []
        for r, l in enumerate(lines):
            if l in line_name_list:
                lines_row_num.append(r)
        
        lines_x_list = []
        lines_y_list = []
        for num in range(line_num):
            line_x  = []
            line_y  = []
            if num != line_num-1:
                for i in lines[lines_row_num[num]+1:lines_row_num[num+1]-1]:
                    line_x.append(i.split()[0])
                    line_y.append(i.split()[1])
            if num == line_num-1:
                for i in lines[lines_row_num[num]+1:len(lines)]:
                    line_x.append(i.split()[0])
                    line_y.append(i.split()[1])
            lines_x_list.append(np.array(line_x, dtype=float))
            lines_y_list.append(np.array(line_y, dtype=float))
    return lines_x_list, lines_y_list

def plot_shear_crack(origin, height, width, fig_num=1):
    x, y = read_crack_data(crack_data='/Crack/DiagonalTensile.txt', line_num=1)
    plt.figure(fig_num)
    plt.plot((origin[0]+x[0])*width, (origin[1]+y[0])*height, '-r')
    plt.plot((origin[0]+x[0])*width, (origin[1]+1-y[0])*height, '-r')
    plt.title('Shear Failure',fontsize=15,fontweight='bold')
    plt.xlabel('Note: Pictures shown are for illustration purpose only.')
    plt.axis('scaled')
    return None

def plot_flexural_crack(origin, height, width, fig_num=1):
    x, y = read_crack_data(crack_data='/Crack/Flexure.txt', line_num=4)
    plt.figure(fig_num)
    for i in range(4):
        plt.plot((origin[0]+x[i])*width, (origin[1]+y[i])*height, '-r')
        plt.plot((origin[0]+1-x[i])*width, (origin[1]+y[i])*height, '-r')
    plt.title('Flexural Failure',fontsize=15,fontweight='bold')
    plt.xlabel('Note: Pictures shown are for illustration purpose only.')
    plt.axis('scaled')
    return None

def plot_flexural_shear_crack(origin, height, width, fig_num=1):
    x, y = read_crack_data(crack_data='/Crack/FlexuralShear.txt', line_num=4)
    plt.figure(fig_num)
    for i in range(4):
        plt.plot((origin[0]+x[i])*width, (origin[1]+y[i])*height, '-r')
        plt.plot((origin[0]+1-x[i])*width, (origin[1]+y[i])*height, '-r')
    plt.title('Flexure-Shear Failure',fontsize=15,fontweight='bold')
    plt.xlabel('Note: Pictures shown are for illustration purpose only.')
    plt.axis('scaled')
    return None

def plot_sliding_crack(origin, height, width, fig_num=1):
    x, y = read_crack_data(crack_data='/Crack/Sliding.txt', line_num=1)
    plt.figure(fig_num)
    plt.plot((origin[0]+x[0])*width, (origin[1]+y[0])*height, '-r', linewidth=8)
    plt.title('Sliding Failure',fontsize=15,fontweight='bold')
    plt.xlabel('Note: Pictures shown are for illustration purpose only.')
    plt.axis('scaled')
    return None

class predictor(QWidget):

    def __init__(self):
        super(predictor, self).__init__()
        self.initUI()
        self.setWindowTitle('Prediction of Failure Modes, Strength and Deformation Capacity of RC Shear Walls through Machine Learning')
        
        # ico_path = sys.path[0]
        # if ico_path.find('base_library.zip') == -1:
        #     ico_path = ico_path + '\data\logo.ico'
        #     print('ico_path: ', ico_path)
        # else:
        #     ico_path = ico_path[0:ico_path.find('base_library.zip')] + 'data\logo.ico'
        #     print('ico_path: ', ico_path)
        # self.setWindowIcon(QtGui.QIcon(ico_path))
        self.setWindowIcon(QtGui.QIcon(sys.path[0]+'/logo.jpg'))

    def initUI(self):
        # subtitle
        self.intro_label = QLabel('Prediction of Failure Modes, Strength and Deformation Capacity\nof RC Shear Walls through Machine Learning')
        self.intro_label.setAlignment(Qt.AlignCenter)
        self.intro_label.setStyleSheet('color:rgb(0,150,195); font-weight:bold; background-color:orange; border-radius:10px; border:2px groove gray; border-style:outset')

        section_type_list = ['Rectangular', 'Barbell', 'Flange']
        # parameters
        self.capacity_ratio_label = QLabel('V<sub>n</sub> / V<sub>s</sub>', self)
        self.capacity_ratio_label.setStyleSheet('font-style:italic')
        self.shear_span_label     = QLabel('M / (Vl<sub>w</sub>)', self)
        self.shear_span_label.setStyleSheet('font-style:italic')
        self.axial_ratio_label    = QLabel('P / (f<sub>c</sub>A<sub>g</sub>)', self)
        self.axial_ratio_label.setStyleSheet('font-style:italic')
        self.longi_reinf_label    = QLabel('ρ<sub>vb</sub>f<sub>y,vb</sub> / f<sub>c</sub>', self)
        self.longi_reinf_label.setStyleSheet('font-style:italic')
        self.hoop_reinf_label     = QLabel('ρ<sub>hb</sub>f<sub>y,hb</sub> / f<sub>c</sub>', self)
        self.hoop_reinf_label.setStyleSheet('font-style:italic')
        self.width_to_thick_label = QLabel('l<sub>w</sub> / t<sub>w</sub>', self)
        self.width_to_thick_label.setStyleSheet('font-style:italic')
        self.web_hor_reinf_label  = QLabel('ρ<sub>hw</sub>f<sub>y,vw</sub> / f<sub>c</sub>', self)
        self.web_hor_reinf_label.setStyleSheet('font-style:italic')
        self.web_ver_reinf_label  = QLabel('ρ<sub>vw</sub>f<sub>y,vw</sub> / f<sub>c</sub>', self)
        self.web_ver_reinf_label.setStyleSheet('font-style:italic')
        self.Ab_Ag_label          = QLabel('A<sub>b</sub> / A<sub>g</sub>', self)
        self.Ab_Ag_label.setStyleSheet('font-style:italic')
        self.section_type_label   = QLabel('Section type')
        
        self.failure_mode_label   = QLabel('Failure Mode', self)
        self.failure_mode_label.setStyleSheet('font-weight:bold')
        self.strength_label       = QLabel('V / (A<sub>g</sub>f<sub>c</sub>)', self)
        self.strength_label.setStyleSheet('font-style:italic; font-weight:bold')
        self.deformation_label    = QLabel('θ<sub>u</sub> (%)', self)
        self.deformation_label.setStyleSheet('font-style:italic; font-weight:bold')

        # input box
        self.capacity_ratio_line = QLineEdit('1.45', self)
        self.capacity_ratio_line.setStyleSheet('background-color:white; border-radius:10px')
        self.shear_span_line     = QLineEdit('1.47', self)
        self.shear_span_line.setStyleSheet('background-color:white; border-radius:10px')
        self.shear_span_line.editingFinished.connect(self.on_section_type_change)
        self.axial_ratio_line    = QLineEdit('0.08', self)
        self.axial_ratio_line.setStyleSheet('background-color:white; border-radius:10px')
        self.longi_reinf_line    = QLineEdit('0.48', self)
        self.longi_reinf_line.setStyleSheet('background-color:white; border-radius:10px')
        self.hoop_reinf_line     = QLineEdit('0.10', self)
        self.hoop_reinf_line.setStyleSheet('background-color:white; border-radius:10px')
        self.width_to_thick_line = QLineEdit('12.96', self)
        self.width_to_thick_line.setStyleSheet('background-color:white; border-radius:10px')
        self.width_to_thick_line.editingFinished.connect(self.on_section_type_change)
        self.web_hor_reinf_line  = QLineEdit('0.08', self)
        self.web_hor_reinf_line.setStyleSheet('background-color:white; border-radius:10px')
        self.web_ver_reinf_line  = QLineEdit('0.08', self)
        self.web_ver_reinf_line.setStyleSheet('background-color:white; border-radius:10px')
        self.Ab_Ag_line          = QLineEdit('0.09', self)
        self.Ab_Ag_line.setStyleSheet('background-color:white; border-radius:10px')
        self.section_type_combox = QComboBox(self)
        self.section_type_combox.setStyleSheet('background-color:white; border-radius:10px')
        self.section_type_combox.addItems(section_type_list)
        self.section_type_combox.currentTextChanged.connect(self.on_section_type_change)
        self.failure_mode_line   = QLineEdit(self)
        self.failure_mode_line.setStyleSheet('background-color:white; border-radius:10px')
        self.strength_line       = QLineEdit(self)
        self.strength_line.setStyleSheet('background-color:white; border-radius:10px')
        self.deformation_line    = QLineEdit(self)
        self.deformation_line.setStyleSheet('background-color:white; border-radius:10px')

        # button
        self.pred_button = QPushButton('Predict', self)
        self.pred_button.clicked.connect(self.on_pred_button_click)
        self.pred_button.setStyleSheet('color:red; background-color:rgb(0,150,195); border-radius:10px; border:2px groove gray; border-style:outset;')

        # plot
        self.fig_fm = plt.figure(1, figsize=(5,10))
        self.plt_fm = FigureCanvas(self.fig_fm)
        plot_wall(section_type='Rectangular', height=2, width=1, thickness=0.1, fig_num=1)
        self.plt_fm.draw()

        # self.note_label = QLabel('Note: Pictures shown are for illustration purpose only.', self)
        
        # layout
        self.grid_layout_intro  = QGridLayout()
        self.grid_layout_input  = QGridLayout()
        self.grid_layout_output = QGridLayout()
        self.grid_layout_plot   = QGridLayout()
        self.grid_layout_pred   = QGridLayout()
        
        self.grid_layout_intro.addWidget(self.intro_label, 0, 0, 1, 3)
        self.row_start = 2
        self.grid_layout_input.addWidget(self.capacity_ratio_label, self.row_start+0, 0, 1, 1)
        self.grid_layout_input.addWidget(self.capacity_ratio_line, self.row_start+0, 1, 1, 1)
        self.grid_layout_input.addWidget(self.shear_span_label, self.row_start+1, 0, 1, 1)
        self.grid_layout_input.addWidget(self.shear_span_line, self.row_start+1, 1, 1, 1)
        self.grid_layout_input.addWidget(self.axial_ratio_label, self.row_start+2, 0, 1, 1)
        self.grid_layout_input.addWidget(self.axial_ratio_line, self.row_start+2, 1, 1, 1)
        self.grid_layout_input.addWidget(self.longi_reinf_label, self.row_start+3, 0, 1, 1)
        self.grid_layout_input.addWidget(self.longi_reinf_line, self.row_start+3, 1, 1, 1)
        self.grid_layout_input.addWidget(self.hoop_reinf_label, self.row_start+4, 0, 1, 1)
        self.grid_layout_input.addWidget(self.hoop_reinf_line, self.row_start+4, 1, 1, 1)
        self.grid_layout_input.addWidget(self.width_to_thick_label, self.row_start+5, 0, 1, 1)
        self.grid_layout_input.addWidget(self.width_to_thick_line, self.row_start+5, 1, 1, 1)
        self.grid_layout_input.addWidget(self.web_hor_reinf_label, self.row_start+6, 0, 1, 1)
        self.grid_layout_input.addWidget(self.web_hor_reinf_line, self.row_start+6, 1, 1, 1)
        self.grid_layout_input.addWidget(self.web_ver_reinf_label, self.row_start+7, 0, 1, 1)
        self.grid_layout_input.addWidget(self.web_ver_reinf_line, self.row_start+7, 1, 1, 1)
        self.grid_layout_input.addWidget(self.Ab_Ag_label, self.row_start+8, 0, 1, 1)
        self.grid_layout_input.addWidget(self.Ab_Ag_line, self.row_start+8, 1, 1, 1)
        self.grid_layout_input.addWidget(self.section_type_label, self.row_start+9, 0, 1, 1)
        self.grid_layout_input.addWidget(self.section_type_combox, self.row_start+9, 1, 1, 1)
        self.grid_layout_input.setSpacing(17)
        # self.row_start = -11
        self.grid_layout_output.addWidget(self.failure_mode_label, self.row_start+11, 0, 1, 1)
        self.grid_layout_output.addWidget(self.failure_mode_line, self.row_start+11, 1, 1, 1)
        self.grid_layout_output.addWidget(self.strength_label, self.row_start+12, 0, 1, 1)
        self.grid_layout_output.addWidget(self.strength_line, self.row_start+12, 1, 1, 1)
        self.grid_layout_output.addWidget(self.deformation_label, self.row_start+13, 0, 1, 1)
        self.grid_layout_output.addWidget(self.deformation_line, self.row_start+13, 1, 1, 1)
        self.grid_layout_output.setSpacing(17)
        # self.row_start = 0
        self.grid_layout_plot.addWidget(self.plt_fm, self.row_start+0, 1, 16, 1)
        # self.row_start = -15
        self.grid_layout_pred.addWidget(self.pred_button, self.row_start+15, 0, 1, 2)
        # self.grid_layout.setSpacing(20)

        self.groupbox_input  = QGroupBox('Input variables', self)
        self.groupbox_input.setStyleSheet('QGroupBox:title {color: rgb(0,150,195);}')
        self.groupbox_input.setLayout(self.grid_layout_input)
        self.groupbox_output = QGroupBox('Output variables', self)
        self.groupbox_output.setStyleSheet('QGroupBox:title {color: rgb(0,150,195);}')
        self.groupbox_output.setLayout(self.grid_layout_output)
        self.vbox_layout_var = QVBoxLayout()
        self.hbox_layout_var_plot = QHBoxLayout()
        self.vbox_layout_all = QVBoxLayout()
        self.vbox_layout_var.addWidget(self.groupbox_input)
        self.vbox_layout_var.addWidget(self.groupbox_output)
        self.vbox_layout_var.addLayout(self.grid_layout_pred)
        self.hbox_layout_var_plot.addLayout(self.vbox_layout_var)
        self.hbox_layout_var_plot.addLayout(self.grid_layout_plot)
        self.vbox_layout_all.addLayout(self.grid_layout_intro)
        self.vbox_layout_all.addLayout(self.hbox_layout_var_plot)

        self.setLayout(self.vbox_layout_all)
    
    @pyqtSlot()
    def on_section_type_change(self):
        self.capacity_ratio_val = float(self.capacity_ratio_line.text())
        self.shear_span_val     = float(self.shear_span_line.text())
        self.axial_ratio_val    = float(self.axial_ratio_line.text())
        self.longi_reinf_val    = float(self.longi_reinf_line.text())
        self.hoop_reinf_val     = float(self.hoop_reinf_line.text())
        self.width_to_thick_val = float(self.width_to_thick_line.text())
        self.web_hor_reinf_val  = float(self.web_hor_reinf_line.text())
        self.web_ver_reinf_val  = float(self.web_ver_reinf_line.text())
        self.Ab_Ag_val          = float(self.Ab_Ag_line.text())
        self.section_type_val   = self.section_type_combox.currentText()

        self.fig_fm.clf()
        width     = 1
        height    = self.shear_span_val*width
        thickness = width/self.width_to_thick_val
        plot_wall(section_type=self.section_type_val, height=height, width=1, thickness=thickness, fig_num=1)
        self.plt_fm.draw()

    @pyqtSlot()
    def on_pred_button_click(self):
        self.capacity_ratio_val = float(self.capacity_ratio_line.text())
        self.shear_span_val     = float(self.shear_span_line.text())
        self.axial_ratio_val    = float(self.axial_ratio_line.text())
        self.longi_reinf_val    = float(self.longi_reinf_line.text())
        self.hoop_reinf_val     = float(self.hoop_reinf_line.text())
        self.width_to_thick_val = float(self.width_to_thick_line.text())
        self.web_hor_reinf_val  = float(self.web_hor_reinf_line.text())
        self.web_ver_reinf_val  = float(self.web_ver_reinf_line.text())
        self.Ab_Ag_val          = float(self.Ab_Ag_line.text())
        self.section_type_val   = self.section_type_combox.currentText()
        # predict failure mode
        failure_mode_display = predictor_fm(capacity_ratio=self.capacity_ratio_val, shear_span=self.shear_span_val,
                    axial_ratio=self.axial_ratio_val, longi_reinf=self.longi_reinf_val,
                    hoop_reinf=self.hoop_reinf_val, width_to_thick=self.width_to_thick_val,
                    web_hor_reinf=self.web_hor_reinf_val, web_ver_reinf=self.web_ver_reinf_val,
                    Ab_Ag=self.Ab_Ag_val, section_type=self.section_type_val)
        self.failure_mode_line.setText(failure_mode_display)
        # predict strength capacity
        strength_dispaly = predictor_strength(shear_span=self.shear_span_val, axial_ratio=self.axial_ratio_val,
                    longi_reinf=self.longi_reinf_val, hoop_reinf=self.hoop_reinf_val,
                    width_to_thick=self.width_to_thick_val, web_hor_reinf=self.web_hor_reinf_val,
                    web_ver_reinf=self.web_ver_reinf_val, Ab_Ag=self.Ab_Ag_val,
                    section_type=self.section_type_val)
        self.strength_line.setText('{0:4f}'.format(strength_dispaly))
        # predict deformation capacity
        deformation_dispaly = predictor_deformation(shear_span=self.shear_span_val, axial_ratio=self.axial_ratio_val,
                    longi_reinf=self.longi_reinf_val, hoop_reinf=self.hoop_reinf_val,
                    width_to_thick=self.width_to_thick_val, web_hor_reinf=self.web_hor_reinf_val,
                    web_ver_reinf=self.web_ver_reinf_val, Ab_Ag=self.Ab_Ag_val,
                    section_type=self.section_type_val)
        self.deformation_line.setText('{0:4f}'.format(deformation_dispaly))
        # plot crack
        self.fig_fm.clf()
        plt.cla()
        width     = 1
        height    = self.shear_span_val*width
        thickness = width/self.width_to_thick_val
        plot_wall(section_type=self.section_type_val, height=height, width=1, thickness=thickness, fig_num=1)
        if failure_mode_display == 'Shear':
            plot_shear_crack(origin=[0,0], height=height, width=width)
            self.plt_fm.draw()
        if failure_mode_display == 'Flexure':
            plot_flexural_crack(origin=[0,0], height=height, width=width)
            self.plt_fm.draw()
        if failure_mode_display == 'Flexure-Shear':
            plot_flexural_shear_crack(origin=[0,0], height=height, width=width)
            self.plt_fm.draw()
        if failure_mode_display == 'Sliding':
            plot_sliding_crack(origin=[0,0], height=height, width=width)
            self.plt_fm.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QtGui.QFont()
    # font.setFamily("Times New Roman") # 字体
    font.setFamily("Arial") # 字体
    font.setPointSize(15)   # 字体大小
    app.setFont(font)
    demo = predictor()
    demo.show()
    sys.exit(app.exec_())