"""Explain selection, sizing, and duplicate suppression without account balances."""
from collections import Counter


STATUS_LABELS = {
    'skipped_already_executed': '같은 거래일·신호의 이전 실행으로 중복 주문 차단',
    'orders_submitted': '주문 전송 완료 (체결 확인과 별도)',
    'order_submission_failed': '주문 전송 실패 또는 중단',
    'no_orders_sizing_limits': '종목 선정 완료, 최소 주문금액·1주 조건으로 주문 없음',
    'no_orders_at_target': '보유 수량이 목표와 일치하여 주문 없음',
    'no_orders_recheck': '주문 전 재검사에서 전부 제외',
    'no_orders': '실행 가능한 주문 없음',
}


def execution_diagnostics(targets, decisions, raw_count, final_count, execution, min_trade):
    reasons = Counter(row.get('reason', 'unknown') for row in decisions)
    selected = sum(1 for ticker, weight in targets.items() if ticker != '__CASH__' and weight > 0)
    sizing = sum(reasons[key] for key in (
        'new_position_below_minimum_or_one_share', 'target_below_minimum'))
    if execution.get('status') == 'skipped_already_executed':
        status = 'skipped_already_executed'
    elif execution.get('status') in {'failed', 'failed_exception'} or execution.get('failed_orders', 0) or execution.get('abort_reason'):
        status = 'order_submission_failed'
    elif execution.get('executed_orders', 0):
        status = 'orders_submitted'
    elif raw_count and not final_count:
        status = 'no_orders_recheck'
    elif not raw_count and sizing:
        status = 'no_orders_sizing_limits'
    elif not raw_count and reasons['quantity_already_at_target']:
        status = 'no_orders_at_target'
    else:
        status = 'no_orders'
    return dict(status=status, selected_security_count=selected,
        target_cash_weight=float(targets.get('__CASH__', 0)),
        min_trade_value=min_trade, raw_plan_count=raw_count, final_plan_count=final_count,
        sizing_excluded_count=sizing, planning_reason_counts=dict(reasons),
        submitted_order_count=int(execution.get('executed_orders', 0)))


def summary_markdown(payload):
    d = payload['diagnostics']
    lines = [
        '### 종목 선정·주문 실행 결과', '',
        f"- 거래일: {payload['trade_date']} / 신호일: {payload['signal_date']}",
        f"- 결과: {STATUS_LABELS[d['status']]}",
        f"- 선정 종목: {d['selected_security_count']}개 / 목표 현금: {d['target_cash_weight']:.1%}",
        f"- 최소 주문금액: {d['min_trade_value']:,.0f}원" if d['min_trade_value'] is not None else '- 최소 주문금액: 미기록',
        f"- 주문계획: {d['raw_plan_count']}건 / 재검사 통과: {d['final_plan_count']}건 / 전송: {d['submitted_order_count']}건",
        f"- 최소금액·1주 조건 제외: {d['sizing_excluded_count']}개",
    ]
    prior = payload.get('execution', {}).get('prior_execution')
    if prior:
        lines += [f"- 이전 실행: {prior.get('run_id')} / 시각: {prior.get('timestamp')}",
                  f"- 이전 주문 전송: {prior.get('submitted_order_count', 0)}건"]
        if prior.get('github_run_id'):
            lines.append(f"- 이전 GitHub 실행 ID: {prior['github_run_id']}")
    if d['status'] == 'no_orders_sizing_limits':
        lines.append('- 중복 실행 차단이 아닙니다. 배정금액과 주문 단위 조건을 확인하세요.')
    return '\n'.join(lines) + '\n\n'
