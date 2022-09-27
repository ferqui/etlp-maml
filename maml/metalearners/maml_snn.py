import torch
import torch.nn.functional as F
import numpy as np

from torchmeta.utils import gradient_update_parameters

from maml.metalearners.maml import ModelAgnosticMetaLearning
from maml.utils import tensors_to_device, compute_accuracy

__all__ = ['MetaETLP']


class MetaETLP(ModelAgnosticMetaLearning):

    def get_outer_loss(self, batch):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
                'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
            })

        mean_outer_loss = torch.tensor(0., device=self.device)
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            params, adaptation_results = self.adapt(train_inputs, train_targets,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                test_logits=0
                self.model.reset()
                for t in range(test_inputs.shape[1]):
                    test_logits += self.model(test_inputs[:,t], params=params)
                outer_loss = self.loss_function(test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy(
                    test_logits, test_targets)

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results

    def adapt(self, inputs, targets, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False):
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
        params = None

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            logits = 0
            self.model.reset()
            for t in range(inputs.shape[1]):
                logits += self.model(inputs[:,t], params=params)
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()

            if (step == 0) and is_classification_task:
                results['accuracy_before'] = compute_accuracy(logits, targets)

            self.model.zero_grad()
            params = gradient_update_parameters(self.model, inner_loss,
                step_size=step_size, params=params,
                first_order=(not self.model.training) or first_order)

        return params, results