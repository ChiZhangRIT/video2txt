import os, shutil, random, csv, cv2, numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from models.c3d import C3D

class Dataset():
    phases = ['train', 'test', 'val']
    has_shots = False
    has_raw_frames = False
    descriptors_step = 5

    def __init__(self):
        self.tokenizer = Tokenizer(self.vocabulary_size)
        all_captions = []
        for phase in self.phases:
            _, captions = self.get_annotations(phase)
            all_captions.extend(captions)
        self.tokenizer.fit_on_texts(all_captions)

    def get_annotations(self, phase):
        raise NotImplementedError

    def load_snippet(self, snippet_name):
        resnet_descriptor = np.load(self.features_dir + '/' + snippet_name.split('/')[-1] + '_resnet.npy')
        c3d_descriptor = np.load(self.features_dir + '/' + snippet_name.split('/')[-1] + '_c3d.npy')

        if self.has_shots:
            shot_indexes = np.ceil(
                np.loadtxt(self.features_dir + '/' + snippet_name.split('/')[-1] + '_shots.txt').reshape((-1, 2))[:,
                0] / self.descriptors_step).astype(int)

            if resnet_descriptor.shape[0] > c3d_descriptor.shape[0]:
                resnet_descriptor = resnet_descriptor[:c3d_descriptor.shape[0]]
                shots = np.zeros((c3d_descriptor.shape[0], 1))
            else:
                c3d_descriptor = c3d_descriptor[:resnet_descriptor.shape[0]]
                shots = np.zeros((resnet_descriptor.shape[0], 1))

            shot_indexes[np.where(shot_indexes >= resnet_descriptor.shape[0])] = 0
            shots[np.ix_(shot_indexes, np.zeros((shot_indexes.shape[0])).astype(int))] = 1
            return [np.concatenate([resnet_descriptor, c3d_descriptor], axis=-1), shots]
        else:
            if resnet_descriptor.shape[0] > c3d_descriptor.shape[0]:
                resnet_descriptor = resnet_descriptor[:c3d_descriptor.shape[0]]
            else:
                c3d_descriptor = c3d_descriptor[:resnet_descriptor.shape[0]]
            return np.concatenate([resnet_descriptor, c3d_descriptor], axis=-1)

    def generator(self, b_s, phase, show_ids=False, with_shots=False):
        assert (phase in self.phases)
        if (with_shots and not self.has_shots):
            raise AttributeError

        snippet_ids, captions = self.get_annotations(phase)
        captions = self.tokenizer.texts_to_sequences(captions)

        while True:
            if phase == 'train':
                combined = zip(snippet_ids, captions)
                random.shuffle(combined)
                snippet_ids, captions = zip(*combined)
            count = 0
            while count < len(captions):
                X = np.zeros((b_s, self.max_video_len, 2048 + 4096))
                X_shots = np.zeros((b_s, self.max_video_len, 1))
                Y = np.zeros((b_s, self.max_caption_len), dtype=np.int32)
                this_snippet_ids = []
                i = 0
                j = 0
                while i < b_s:
                    if self.has_shots:
                        try:
                            [this_X, this_X_shots] = self.load_snippet(snippet_ids[(count + j) % len(captions)])
                            X[i, :this_X.shape[0]] = this_X[:self.max_video_len]
                            X_shots[i, :this_X_shots.shape[0]] = this_X_shots[:self.max_video_len]
                            caption = captions[(count + j) % len(captions)]
                            Y[i, :len(caption)] = caption
                            this_snippet_ids.append(snippet_ids[(count + j) % len(captions)])
                            i += 1
                        except:
                            pass
                    else:
                        try:
                            this_X = self.load_snippet(snippet_ids[(count + j) % len(captions)])
                            X[i, :this_X.shape[0]] = this_X[:self.max_video_len]
                            caption = captions[(count + j) % len(captions)]
                            Y[i, :len(caption)] = caption
                            this_snippet_ids.append(snippet_ids[(count + j) % len(captions)])
                            i += 1
                        except:
                            pass
                    j += 1

                if with_shots:
                    all_X = [X, X_shots, Y]
                else:
                    all_X = [X, Y]

                if show_ids:
                    yield all_X, np.expand_dims(Y, -1).copy(), this_snippet_ids
                else:
                    yield all_X, np.expand_dims(Y, -1).copy()
                count += j

    def compute_resnet_descriptors(self):
        b_s = 100
        base_model = ResNet50(weights='imagenet')
        model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)

        snippets = set()
        for phase in self.phases:
            snippets.update(self.get_annotations(phase)[0])

        for snippet_i, snippet in enumerate(snippets):
            if not self.has_raw_frames:
                video = cv2.VideoCapture(self.videos_dir + '/' + snippet + '.avi')
            frame_count = 0
            counter = 1
            stop = False
            print "Processing snippet %s (%d/%d)" % (snippet, snippet_i, len(snippets))
            snippet_descriptors = np.zeros((999 * b_s, 2048))
            for b in range(999):
                batch = np.zeros((b_s, 3, 224, 224))
                for t in range(b_s):
                    if self.has_raw_frames:
                        frame = cv2.imread(self.videos_dir + '/' + snippet + '/%04d.jpg' % counter)
                        retval = frame is not None
                        counter += 1
                    else:
                        retval, frame = video.read()
                    if retval:
                        batch[t] = cv2.resize(frame, (224, 224)).transpose((2, 0, 1))
                        frame_count += 1
                    else:
                        stop = True
                        break
                    for _ in range(self.descriptors_step - 1):
                        if self.has_raw_frames:
                            counter += 1
                        else:
                            video.read()
                batch[:, 0] -= 103.939
                batch[:, 1] -= 116.779
                batch[:, 2] -= 123.68
                snippet_descriptors[b * b_s:(b + 1) * b_s] = model.predict_on_batch(batch).reshape(batch.shape[0], -1)
                if stop:
                    break

            np.save(self.features_dir + '/' + snippet.split('/')[-1] + '_resnet.npy', snippet_descriptors[:frame_count])

    def compute_c3d_descriptors(self):
        b_s = 5
        base_model = C3D()
        model = Model(input=base_model.input, output=base_model.get_layer('fc7').output)

        snippets = set()
        for phase in self.phases:
            snippets.update(self.get_annotations(phase)[0])

        for snippet_i, snippet in enumerate(snippets):
            if not self.has_raw_frames:
                video = cv2.VideoCapture(self.videos_dir + '/' + snippet + '.avi')
            frame_count = 0
            counter = 1
            stop = False
            print "Processing snippet %s (%d/%d)" % (snippet, snippet_i, len(snippets))
            snippet_descriptors = np.zeros((999 * b_s, 4096))
            first_batch = True
            for b in range(999):
                batch = np.zeros((b_s, 3, 16, 112, 112))
                for t in range(b_s):
                    if first_batch:
                        be = 0
                        r = 16
                        first_batch = False
                    else:
                        be = 16 - self.descriptors_step
                        r = 16
                    for tt in range(be, r):
                        if self.has_raw_frames:
                            frame = cv2.imread(self.videos_dir + '/' + snippet + '/%04d.jpg' % counter)
                            retval = frame is not None
                            counter += 1
                        else:
                            retval, frame = video.read()
                        if retval:
                            batch[t, :, tt] = cv2.resize(frame, (112, 112)).transpose((2, 0, 1))
                        else:
                            stop = True
                            break
                    frame_count += 1
                    if stop:
                        break
                    if t < b_s - 1:
                        batch[t + 1, :, :16 - self.descriptors_step] = batch[t, :, -16 + self.descriptors_step:]

                snippet_descriptors[b * b_s:(b + 1) * b_s] = model.predict_on_batch(batch).reshape(batch.shape[0], -1)
                if stop:
                    break

            np.save(self.features_dir + '/' + snippet.split('/')[-1] + '_c3d.npy', snippet_descriptors[:frame_count])

    def compute_shots(self):
        snippets = set()
        for phase in self.phases:
            snippets.update(self.get_annotations(phase)[0])

        for snippet_i, snippet in enumerate(snippets):
            video_path = self.videos_dir + '/' + snippet + '.avi'
            print "%s (%d/%d)" % (video_path, snippet_i, len(snippets))
            os.system('/raid/lbaraldi/shot-detector/build/ShotDetector ' + video_path + ' 3 > /dev/null 2>&1')
            shutil.move(video_path[:-4] + '_shots.txt', self.features_dir + '/' + snippet.split('/')[-1] + '_shots.txt')
            os.remove(video_path[:-4] + '_trans.txt')
            os.remove(video_path[:-4] + '_diffs.txt')

class MVAD(Dataset):
    base_dir = 'datasets/M-VAD'
    annotation_dir = base_dir+'/annotations'
    videos_dir = base_dir+'/videos'
    features_dir = base_dir+'/features'
    vocabulary_size = 6090
    nb_train_samples = 36921
    nb_val_samples = 4651
    nb_test_samples = 4951
    max_caption_len = 20
    max_video_len = 100
    embedding_size = 256
    has_shots = True

    def get_annotations(self, phase):
        assert (phase in self.phases)
        snippets = []
        captions = []

        for film_srt in os.listdir(self.annotation_dir + '/' + phase):
            for line_i, line in enumerate(open(self.annotation_dir + '/' + phase + '/' + film_srt, 'r')):
                if line_i % 4 == 0:
                    snippets.append(film_srt[:-4] + '/video/' + line.strip())
                elif line_i % 4 == 2:
                    seq = text_to_word_sequence(line.strip())[:self.max_caption_len - 2]
                    captions.append('STARTTOKEN ' + ' '.join(seq) + ' STOPTOKEN')

        return snippets, captions

class MPII_MD(Dataset):
    base_dir = 'datasets/MPII-MD'
    annotation_dir = base_dir + '/annotations'
    videos_dir = base_dir + '/jpgAllFrames'
    features_dir = base_dir + '/features'
    vocabulary_size = 7198
    nb_train_samples = 56861
    nb_val_samples = 4930
    nb_test_samples = 6584
    max_caption_len = 20
    max_video_len = 60  # da sistemare
    embedding_size = 256
    has_raw_frames = True

    def get_annotations(self, phase):
        assert (phase in self.phases)
        snippets = []
        captions = []

        snippet_prefixes = []
        for line in open(self.annotation_dir+'/dataSplit.txt', 'r'):
            if line.split('\t')[1].strip().startswith(phase):
                snippet_prefixes.append(line.split('\t')[0].strip())

        for line in open(self.annotation_dir+'/annotations-someone.csv', 'r'):
            if any(line.split('\t')[0].strip().startswith(p) for p in snippet_prefixes):
                snippet_name = '_'.join(line.split('\t')[0].split('_')[:-1]) + '/' + line.split('\t')[0]
                snippets.append(snippet_name)
                seq = text_to_word_sequence(line.split('\t')[1].strip())[:self.max_caption_len - 2]
                captions.append('STARTTOKEN ' + ' '.join(seq) + ' STOPTOKEN')


        return snippets, captions

class Youtube2Text(Dataset):
    base_dir = 'datasets/MSR Video Description Corpus'
    csv_file = base_dir+'/MSR Video Description Corpus.csv'
    videos_dir = base_dir+'/YouTubeClips'
    features_dir = base_dir+'/features'
    vocabulary_size = 2515
    max_caption_len = 44
    nb_train_samples = 65591
    nb_val_samples = 5614
    nb_test_samples = 14345
    max_video_len = 100
    embedding_size = 512

    def get_annotations(self, phase):
        snippets = []
        captions = []
        videos = []
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                row = [x.decode('utf8') for x in row]
                if i > 0 and row[6] == 'English':# and row[4] == 'clean':
                    videos.append(row[0])
                    snippets.append(row[0] + '_' + row[1] + '_' + row[2])
                    captions.append('STARTTOKEN ' + row[-1].encode('ascii', 'ignore')[:self.max_caption_len-2] + ' STOPTOKEN')

        if phase == 'train':
            valid_videos = sorted(set(videos))[:1200]
        elif phase == 'val':
            valid_videos = sorted(set(videos))[1200:1300]
        else:
            valid_videos = sorted(set(videos))[1300:]
        snippets = [snippets[i] for i in range(len(videos)) if videos[i] in valid_videos]
        captions = [captions[i] for i in range(len(videos)) if videos[i] in valid_videos]
        return snippets, captions