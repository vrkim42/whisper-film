import sys
import os
import time
import hashlib
import json
import traceback
import platform
import ctypes.util

# 针对 Windows 环境修复 find_library('c') 返回 None 的问题
if platform.system() == "Windows":
    original_find_library = ctypes.util.find_library
    def patched_find_library(name):
        result = original_find_library(name)
        if result is None and name == "c":
            return "msvcrt.dll"
        return result
    ctypes.util.find_library = patched_find_library

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    import whisper
    print(f"导入的 whisper 模块路径: {whisper.__file__}")
    print("Python 搜索路径:")
    for path in sys.path:
        print(path)
except ImportError as e:
    print(f"导入 whisper 库时出错: {e}")
    print("请确保你已经安装了 whisper 库，可以使用以下命令安装: pip install git+https://github.com/openai/whisper.git")
    raise
except Exception as e:
    print(f"导入 whisper 库时发生未知错误: {e}")
    print(traceback.format_exc())
    raise

# 初始化应用
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置参数
app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'SUBTITLES_FOLDER': 'subtitles',
    'ALLOWED_EXTENSIONS': {'mp4', 'mov', 'avi', 'mkv', 'webm'},
    'MAX_CONTENT_LENGTH': 1024 * 1024 * 1024,  # 1GB 限制
    'MODEL_SIZE': 'base'
})

# 初始化目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SUBTITLES_FOLDER'], exist_ok=True)

# 全局加载模型
try:
    model = whisper.load_model(app.config['MODEL_SIZE'])
except Exception as e:
    print(f"加载 whisper 模型时出错: {e}")
    print(traceback.format_exc())
    raise


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': '未选择文件'}), 400

        file = request.files['video']
        if not file or file.filename == '':
            return jsonify({'error': '无效文件'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'}), 400

        # 生成安全文件名
        filename = secure_filename(file.filename)
        
        # 分块读取计算 MD5，支持大文件
        hasher = hashlib.md5()
        while True:
            chunk = file.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
        file_hash = hasher.hexdigest()
        # 重置文件指针，确保后续 file.save 正常工作
        file.seek(0)

        # 保存文件
        save_filename = f"{file_hash}_{filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], save_filename)
        file.save(video_path)

        # 检查文件是否存在
        if not os.path.exists(video_path):
            raise RuntimeError(f"文件保存失败: {video_path} 不存在")

        # 生成字幕
        subtitles = generate_subtitles(video_path)
        return jsonify({
            'filename': subtitles['filename'],
            'video_url': f'/media/{save_filename}',
            'duration': subtitles['duration']
        })

    except Exception as e:
        app.logger.error(f"文件上传或处理时发生错误: {str(e)}")
        return jsonify({'error': f'文件上传或处理时发生错误，请稍后再试。错误详情: {str(e)}'}), 500


def generate_subtitles(video_path):
    try:
        app.logger.info(f"Generating subtitles for video: {video_path}")

        # 生成唯一字幕文件名
        file_hash = os.path.basename(video_path).split('_')[0]
        subtitles_filename = f"{file_hash}.json"
        subtitles_path = os.path.join(app.config['SUBTITLES_FOLDER'], subtitles_filename)

        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            raise RuntimeError(f"视频文件不存在: {video_path}")

        app.logger.info(f"Video file exists: {video_path}")

        # 如果字幕文件不存在则生成，否则直接读取缓存文件
        if not os.path.exists(subtitles_path):
            result = model.transcribe(video_path)
            with open(subtitles_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'segments': result['segments'],
                    'duration': result['segments'][-1]['end'] if result['segments'] else 0
                }, f, ensure_ascii=False)
        else:
            with open(subtitles_path, 'r', encoding='utf-8') as f:
                result = json.load(f)

        app.logger.info(f"Subtitles generated: {subtitles_path}")

        return {
            'filename': subtitles_filename,
            'duration': result['segments'][-1]['end'] if result['segments'] else 0
        }

    except Exception as e:
        raise RuntimeError(f"字幕生成失败: {str(e)}")


@app.route('/media/<filename>')
def serve_media(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@socketio.on('connect')
def handle_connect():
    emit('connection_status', {'status': 'connected'})


@socketio.on('request_subtitles')
def handle_request_subtitles(data):
    try:
        subtitles_path = os.path.join(app.config['SUBTITLES_FOLDER'], data['filename'])
        if not os.path.exists(subtitles_path):
            emit('subtitles_error', {'error': '字幕文件不存在'})
            return

        with open(subtitles_path, 'r', encoding='utf-8') as f:
            subtitles = json.load(f)

        start_time = time.time()
        current_index = 0

        while current_index < len(subtitles['segments']):
            elapsed = time.time() - start_time
            segment = subtitles['segments'][current_index]

            if elapsed >= segment['start']:
                emit('new_subtitle', {
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['end'] - segment['start']
                })
                current_index += 1
            else:
                socketio.sleep(0.1)

    except Exception as e:
        emit('subtitles_error', {'error': str(e)})


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=8848, debug=True, allow_unsafe_werkzeug=True)