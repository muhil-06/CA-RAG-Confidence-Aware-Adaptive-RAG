from src.pipeline import CARAGPipeline

if __name__ == '__main__':
    p = CARAGPipeline()
    res = p.run('What is the capital of Japan?')
    print('ANSWER:\n', res.answer)
    print('\nMETA:')
    print('retrieved=', res.retrieved)
    print('confidence=', res.confidence)
    print('latency_ms=', res.latency_ms)
    print('estimated_cost_usd=', res.estimated_cost_usd)
