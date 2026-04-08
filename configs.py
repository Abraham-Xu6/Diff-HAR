import os

def get_config(name):
    """根据名称获取数据集配置"""
    if name.upper() == 'PAMAP2':
        return PAMAP2_CONFIG
    elif name.upper() == 'USC-HAD':
        return USC_HAD_CONFIG
    elif name.upper() == 'MMWAVE':
        return MMWAVE_CONFIG
    else:
        raise ValueError(f"未找到名为 '{name}' 的数据集配置")

# =================================================================================
#                           PAMAP2 数据集配置
# =================================================================================
PAMAP2_CONFIG = {
    'name': 'PAMAP2',
    # 修复：改为相对路径，要求用户将数据集解压至项目根目录的 dataset 文件夹下
    'raw_data_dir': "./dataset/PAMAP2/Protocol",
    'preprocessed_data_dir': './data_preprocessed/PAMAP2',
    'results_dir': './results_GZSL_Feature/PAMAP2',
    'window_size': 171,
    'overlap_rate': 0.5,
    'total_classes': 12,
    'num_seen_classes': 10,
    'label_text_dict': {
        0: "Complete rest position with no movement. Characterized by lying flat on the back or side, eyes closed, relaxed muscles, and a state of relaxation.",
        1: "Static posture with minimal movement. Characterized by a seated position where the weight is supported by a surface, legs are bent at the knees, and feet are on the ground or elevated.",
        2: "Static posture with no movement. Characterized by an upright position where the weight is supported by the feet, legs are straight, and arms are relaxed by the sides.",
        3: "Linear movement in a forward direction. Speed is consistent but slower than running. Characterized by alternating movement of left and right legs, swinging arms, and continuous steps.",
        4: "Linear movement in a forward direction. Speed is faster than walking. Characterized by a rapid and forceful stride, both feet leaving the ground simultaneously, and intense arm movement.",
        5: "Linear movement using a bicycle. Speed can vary. Characterized by circular leg movements, continuous pedaling, hand grip on handlebars, and leaning forward or upright posture.",
        6: "Linear movement using poles for support. Speed is consistent but slower than running. Characterized by the use of poles to aid in walking, rhythmic arm swinging, and upright posture.",
        7: "Vertical movement upwards using stairs. Speed is slower than walking on a flat surface. Characterized by lifting legs higher, gripping the railing for support, and continuous step-by-step movement.",
        8: "Vertical movement downwards using stairs. Speed is slower than walking on a flat surface. Characterized by placing the foot on each step cautiously, leaning slightly backward, and using railing for support.",
        9: "Linear movement while cleaning using a vacuum cleaner. Speed is consistent and can vary based on the cleaning area. Characterized by pushing and pulling motions, back and forth movements, and hand grip on the vacuum handle.",
        10: "Static posture with repetitive movement. Characterized by standing in front of an ironing board, moving an iron back and forth over clothes, rotating the wrist, and maintaining a focused posture.",
        11: "Vertical movement upwards and downwards using a jump rope. Speed can vary. Characterized by jumping over a rope that passes under the feet, swinging the rope with wrists, and maintaining a rhythmic jumping pattern."
    }
}

# =================================================================================
#                       USC-HAD 数据集配置
# =================================================================================
USC_HAD_CONFIG = {
    'name': 'USC-HAD',
    # 修复：改为相对路径
    'raw_data_dir': "./dataset/USC-HAD", 
    'preprocessed_data_dir': './data_preprocessed/USC-HAD',
    'results_dir': './results_GZSL_Feature/USC-HAD',
    'window_size': 128,
    'overlap_rate': 0.5, 
    'total_classes': 12,
    'num_seen_classes': 9,
    'label_text_dict': {
        0: "Linear movement in the forward direction. Speed is consistent but slower than running. Characterized by alternating movement of left and right legs.",
        1: "Linear movement to the left. Speed is consistent but slower than running. Characterized by alternating movement of left and right legs.",
        2: "Linear movement to the right. Speed is consistent but slower than running. Characterized by alternating movement of left and right legs.",
        3: "Vertical movement upwards using stairs. Speed is slower than walking on a flat surface. Characterized by lifting legs higher and consistent arm movement for balance.",
        4: "Vertical movement downwards using stairs. Speed is slower than walking on a flat surface. Characterized by placing the foot on each step cautiously and using railing for support.",
        5: "Linear movement in the forward direction. Speed is faster than walking. Characterized by a faster and more forceful stride, with both feet leaving the ground.",
        6: "Vertical movement upwards without forward motion. Characterized by both feet leaving the ground simultaneously and arms usually swinging upwards.",
        7: "Static posture with no movement. Characterized by a seated position where the weight is supported by a surface.",
        8: "Static posture with no movement. Characterized by an upright position where the weight is supported by the feet.",
        9: "Static posture with no movement. Characterized by lying down, eyes closed, and little to no movement.",
        10: "Vertical movement upwards in an elevator. Characterized by a smooth and consistent upward motion without the need for physical exertion.",
        11: "Vertical movement downwards in an elevator. Characterized by a smooth and consistent downward motion without the need for physical exertion."
    }
}

# =================================================================================
#                       mmWave (MMFi) 数据集配置
# =================================================================================
MMWAVE_CONFIG = {
    'name': 'mmWave',
    # 修复：改为相对路径
    'raw_data_dir': "./dataset/MMWAVE", 
    'preprocessed_data_dir': './data_preprocessed/mmWave',
    'results_dir': './results_GZSL_Feature/mmWave',
    'window_size': 150,
    'overlap_rate': 0.2,
    'total_classes': 27,
    'num_seen_classes': 21,
    'label_text_dict': {
        0: "Dynamic movement involving stretching and then relaxing muscles. Characterized by elongation of the body and subsequent release of tension.",
        1: "Dynamic movement expanding the chest horizontally. Characterized by opening the chest and extending the arms sideways.",
        2: "Dynamic movement expanding the chest vertically. Characterized by lifting the arms upwards and expanding the chest.",
        3: "Rotational movement to the left. Characterized by twisting the torso and possibly the hips to the left.",
        4: "Rotational movement to the right. Characterized by twisting the torso and possibly the hips to the right.",
        5: "Dynamic movement involving marching in place. Characterized by lifting and lowering the feet alternately without forward motion.",
        6: "Dynamic movement extending the left limb. Characterized by stretching the left arm or leg outward.",
        7: "Dynamic movement extending the right limb. Characterized by stretching the right arm or leg outward.",
        8: "Dynamic movement lunging forward and to the left. Characterized by stepping forward with the left leg and bending both knees.",
        9: "Dynamic movement lunging forward and to the right. Characterized by stepping forward with the right leg and bending both knees.",
        10: "Dynamic movement extending both limbs simultaneously. Characterized by stretching both arms or legs outward.",
        11: "Dynamic movement lowering the body by bending knees and hips. Characterized by lowering the body towards the ground while keeping the back straight.",
        12: "Dynamic movement raising the left hand upwards. Characterized by lifting the left arm towards the shoulder or above the head.",
        13: "Dynamic movement raising the right hand upwards. Characterized by lifting the right arm towards the shoulder or above the head.",
        14: "Dynamic movement lunging sideways to the left. Characterized by stepping sideways with the left leg and bending the knee.",
        15: "Dynamic movement lunging sideways to the right. Characterized by stepping sideways with the right leg and bending the knee.",
        16: "Dynamic movement waving the left hand. Characterized by moving the left hand back and forth in a waving motion.",
        17: "Dynamic movement waving the right hand. Characterized by moving the right hand back and forth in a waving motion.",
        18: "Dynamic movement bending down to pick up objects. Characterized by lowering the body and reaching towards the ground.",
        19: "Dynamic movement throwing an object to the left side. Characterized by extending the arm and releasing the object.",
        20: "Dynamic movement throwing an object to the right side. Characterized by extending the arm and releasing the object.",
        21: "Dynamic movement kicking towards the left side. Characterized by extending the left leg outward.",
        22: "Dynamic movement kicking towards the right side. Characterized by extending the right leg outward.",
        23: "Dynamic movement extending the left side of the body. Characterized by stretching and lengthening the left side.",
        24: "Dynamic movement extending the right side of the body. Characterized by stretching and lengthening the right side.",
        25: "Vertical movement upwards. Characterized by both feet leaving the ground simultaneously.",
        26: "Dynamic movement bending forward from the waist as a sign of respect or acknowledgment. Characterized by lowering the head and upper body towards the ground."
    }
}