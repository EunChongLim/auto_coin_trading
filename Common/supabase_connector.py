import os
from dotenv import load_dotenv
from supabase import create_client, Client

# 싱글톤 Supabase Client 관리 변수
_supabase_instance = None

def get_singleton_supabase_client():
    """
    Supabase Client를 싱글톤으로 반환합니다.
    여러 번 호출되어도 하나의 객체만 유지됩니다.
    """
    global _supabase_instance
    if _supabase_instance is None:
        load_dotenv()
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        _supabase_instance = create_client(url, key)
    return _supabase_instance

def select_from_table(table_name: str, **kwargs):
    """
    Supabase 테이블에서 데이터를 조회합니다.
    Args:
        table_name (str): 조회할 테이블 이름
        kwargs: 조건 등 추가 파라미터
    Returns:
        실제 조회 결과 반환
    """
    supabase = get_singleton_supabase_client()
    response = (
        supabase.table(table_name)
        .select("*")
        .execute()
    )
    
    return response;

def insert_into_table(table_name: str, data: dict):
    """
    Supabase 테이블에 데이터를 Insert합니다.
    Args:
        table_name (str): 삽입할 테이블 이름
        data (dict): 삽입할 데이터
    Returns:
        실제 insert 결과 반환
    """
    supabase = get_singleton_supabase_client()
    response = (
        supabase.table(table_name)
        .insert(data)
        .execute()
    )
    return response

def update_table(table_name: str, data: dict, match: dict):
    """
    Supabase 테이블의 데이터를 Update합니다.
    Args:
        table_name (str): 업데이트할 테이블 이름
        data (dict): 변경할 데이터
        match (dict): 업데이트할 조건 (예: {'id': 1, 'name': 'test'})
    Returns:
        실제 update 결과 반환
    """
    supabase = get_singleton_supabase_client()
    
    # 쿼리 시작
    query = supabase.table(table_name).update(data)

    # match 조건들을 eq로 체이닝
    for key, value in match.items():
        query = query.eq(key, value)
    
    # 실행
    response = query.execute()
    return response

    # main 함수에서 select_from_table 함수의 정상 동작을 테스트하는 코드를 작성합니다.
if __name__ == "__main__":
    # 테스트를 위한 샘플 테이블명 지정 (예: 'test_table')
    table_name = "tb_model_inf"

    print(f"Supabase '{table_name}' 테이블 select 테스트 시작")

    try:
        response = select_from_table(table_name)
        # response = insert_into_table(table_name, {"model_nm":"model_test1", "model_wt_id":"ecLim"})
        # response = update_table(table_name, {"model_nm":"model_test2"}, {"model_nm":"model_v7"})
        print("조회 결과:")
        print(response)
    except Exception as e:
        print("Supabase select 테스트 중 오류 발생:", e)    
