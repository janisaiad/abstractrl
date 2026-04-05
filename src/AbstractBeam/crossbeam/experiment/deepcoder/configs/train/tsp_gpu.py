from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict(initial_dictionary=dict(
      save_dir='', data_root='',
  ))
  config.domain = 'tsp'
  config.data_gen_timeout = 500
  config.max_num_examples = 5
  config.max_num_inputs = 3
  config.min_num_examples = 2
  config.min_num_inputs = 1
  config.max_search_weight = 24
  config.min_task_weight = 3
  config.max_task_weight = 9
  config.num_tasks_per_weight = 200
  config.skip_probability = 0.0
  config.lambda_skip_probability = 0.0
  config.lambda_fraction = 0.5
  config.shuffle_ops = True
  config.abstraction_refinement = True
  config.data_save_dir = './neurips/tsp/data'
  config.num_datagen_proc = 8
  config.data_gen_seed = 0
  config.num_searches = 50
  config.shard_size = 500
  config.dynamic_time_increase = 50

  config.seed = 42
  config.tout = 3600
  config.io_encoder = 'lambda_signature'
  config.model_type = 'deepcoder'
  config.value_encoder = 'lambda_signature'
  config.grad_accumulate = 4
  config.beam_size = 10
  config.num_proc = 1
  config.gpu_list = '0'
  config.gpu = 0
  config.embed_dim = 128
  config.eval_every = 300
  config.port = '56566'
  config.use_ur = False
  config.do_test = False
  config.timeout = 60.0
  config.restarts_timeout = 10
  config.encode_weight = True
  config.train_steps = 10000000
  config.random_beam = False
  config.use_op_specific_lstm = True
  config.lr = 5e-4
  config.load_model = ''
  config.steps_per_curr_stage = 5000
  config.schedule_type = 'uniform'
  config.json_results_file = './neurips/tsp/results/'
  config.save_dir = './neurips/tsp/models'
  config.synthetic = False

  config.abstraction = True
  config.num_starting_ops = 28
  config.dynamic_tasks = False
  config.use_ur_in_valid = True
  config.initialization_method = 'top'
  config.abstraction_pruning = True
  config.top_k = 2
  config.num_inventions_per_iter = 20
  config.invention_arity = 3
  config.used_invs = None
  config.max_invention = 20
  config.castrate_macros = False
  return config
