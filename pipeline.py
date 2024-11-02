from abc import ABC, abstractmethod
from functools import reduce
from multiprocessing.pool import ThreadPool
from typing import Dict, List

from loguru import logger

from util import measure_time

type PipelineConfig = dict

class Pipeline(ABC):
    @abstractmethod
    def execute(self, config: PipelineConfig, data: any) -> any:
        pass

    def run(self, config: PipelineConfig, data: any) -> any:
        logger.info(f'[{self.__class__.__name__}] Executing')

        with measure_time() as execution_time:
            data = self.execute(config, data)

        logger.info(f'[{self.__class__.__name__}] Executing - Done in {execution_time()}s')
        return data

class PipelineStep(ABC):
    @abstractmethod
    def execute(self, config: PipelineConfig, data: any) -> any:
        pass

    def execute_internal(self, config: PipelineConfig, data: any) -> any:
        logger.info(f'[{self.__class__.__name__}] Executing')

        with measure_time() as execution_time:
            data = self.execute(config, data)

        logger.info(f'[{self.__class__.__name__}] Executing - Done in {execution_time()}s')
        return data

class LinearPipeline(Pipeline):
    steps: List[PipelineStep]

    def __init__(self, steps: List[PipelineStep] = None):
        self.steps = steps or []

    def add_step(self, step: PipelineStep):
        self.steps.append(step)

    def execute(self, config: PipelineConfig, data: any) -> any:
        return reduce(lambda data, step: step.execute_internal(config, data), self.steps, data)
    
class Map(PipelineStep):
    pipeline: Pipeline

    def __init__(self, pipeline: Pipeline | List[PipelineStep], parallel: bool = False, thread_pool_size: int = 4):
        self.pipeline = LinearPipeline(pipeline) if type(pipeline) is list else pipeline
        self.parallel = parallel
        self.thread_pool_size = thread_pool_size

    def execute(self, config: PipelineConfig, data: any) -> any:
        if type(data) is list:
            return self.execute_list(config, data)
        elif type(data) is dict:
            return self.execute_dict(config, data)
        
        raise Exception(f"unsupported type {type(data)}")
    
    def execute_list(self, config: PipelineConfig, data: List[any]) -> List[any]:
        if not self.parallel:
            return [self.pipeline.run(config, filepath) for filepath in data]
        
        logger.debug(f'[Map] mapping list in parallel')

        with ThreadPool(self.thread_pool_size) as thread_pool:
            return thread_pool.map(lambda data: self.pipeline.run(config, data), data)
        
    def execute_dict(self, config: PipelineConfig, data: Dict[any, any]) -> Dict[any, any]:
        if not self.parallel:
            return  { k: self.pipeline.run(config, v) for k, v in data.items() }
        
        logger.debug(f'[Map] mapping dict in parallel')

        with ThreadPool(self.thread_pool_size) as thread_pool:
            result = thread_pool.map(lambda data: self.pipeline.run(config, data), data.values())
            return { k: v for k, v in zip(data.keys(), result) }

class Reduce(PipelineStep):
    pipeline: Pipeline

    def __init__(self, pipeline: Pipeline | PipelineStep | List[PipelineStep]):
        if issubclass(type(pipeline), PipelineStep):
            self.pipeline = LinearPipeline([pipeline])
        elif issubclass(type(pipeline), Pipeline):
            self.pipeline = pipeline
        elif type(pipeline) is list:
            self.pipeline = LinearPipeline(pipeline)
        else:
            raise ValueError(f"unsupported pipeline type {type(pipeline)}")

    def execute(self, config: PipelineConfig, data: any) -> any:
        if type(data) is list:
            return reduce(lambda prev, value: self.pipeline.run(config, (prev, value)), data, None)
        raise Exception(f"unsupported data type {type(data)}")
