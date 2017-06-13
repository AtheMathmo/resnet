'''
Extra dataset wrappers
'''

class AdvExamplesDataset(object):
    '''
    Dataset wrapper for adversarial examples.
    '''

    def __init__(self, examples, labels, targets=None):
        self.examples = examples
        self.labels = labels
        self.targets = targets

        self.num_examples = self.examples.shape[0]

        # Make sure the shapes match
        assert self.num_examples == self.labels.shape[0]
        if self.targets is not None:
            assert self.num_examples == self.targets.shape[0]

    def get_batch_idx(self, idx):
        if self.targets is not None:
            result = {
                "img": self.examples[idx],
                "label": self.labels[idx],
                "target": self.targets[idx]
            }
        else:
            result = {
                "img": self.examples[idx],
                "label": self.labels[idx],
                "target": None
            }
        return result

    def get_size(self):
        return self.num_examples

