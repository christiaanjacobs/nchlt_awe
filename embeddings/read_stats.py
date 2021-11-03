import pickle
import pandas as pd
import argparse


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


def load_record_dict(record_dict_fn):
    print(record_dict_fn)
    with open(record_dict_fn, 'rb') as f:
        record_dict = pickle.load(f)
        data = {'epoch time': [time[1] for time in record_dict['epoch_time']],
                'train loss': [loss[1] for loss in record_dict['train_loss']],
                'AP': [abs(ap[1][-1]) for ap in record_dict['validation_loss']]
                }
        print(data)
#        df = pd.DataFrame(data)
#        df.style.apply(highlight_max)
#        print(df)

def load_options_dict(options_dict_fn):
    print(options_dict_fn)
    with open(options_dict_fn, 'rb') as f:
        options_dict = pickle.load(f)

#        df = pd.DataFrame(options_dict, index=0)
        for key, value in options_dict.items():
            print(str(key)+ ": " +str(value))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rec_dict', type=str)
    parser.add_argument('--opt_dict', type=str) 

    args = parser.parse_args()

    if args.rec_dict is not None:
        load_record_dict(args.rec_dict)

    if args.opt_dict is not None:
        load_options_dict(args.opt_dict)
