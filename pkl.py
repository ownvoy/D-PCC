import pickle


def load_pickle_file(file_path):
    with open(file_path, "rb") as file:  # 'rb'는 바이너리 읽기 모드
        data = pickle.load(file)
        return data


def print_data_sizes(data):
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"Key: {key}, Size: {len(value)}")

    else:
        print("Loaded data is not a dictionary.")


# 파일 경로 지정
file_path = "/workspace/D-PCC/output/semantickitti_train_cube_size_12.pkl"

# 파일 불러오기
loaded_data = load_pickle_file(file_path)

samples = loaded_data[0][0]
print(len(samples.keys()))

print_data_sizes(loaded_data)

