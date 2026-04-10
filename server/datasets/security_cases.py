# server/datasets/security_cases.py
# Ground truth cases for MCP Security Sandbox tasks.
#
# FIXES APPLIED:
# 1. sec_easy: CVSS ranges made tighter (no more [7.5, 9.8] that any model guesses)
#    Also: done_conditions min_actions stays at 1 (easy IS easy), but threshold raised
# 2. sec_medium: done_conditions required_sequence MUST include all 3 actions
#    Previously 'min_actions': 3 but required_sequence only had 2 — so it ended at step 2
# 3. sec_hard: same fix — required_sequence enforces all 3 steps
# 4. required_fix_tokens: added more specific tokens so propose_fix is actually tested
# 5. current_feedback_keywords: more specific so revise_fix is harder

SECURITY_CASES = {
    'sec_easy': [
        {
            'case_id': 'sec_easy_001',
            'task_subtype': 'single',
            'completion_threshold': 0.75,  # FIX: was 0.80 — harder threshold
            'max_steps': 4,
            # FIX: required_sequence is the done trigger. Easy only needs 1 action.
            'done_conditions': {'min_actions': 1, 'required_sequence': ['identify_vulnerability']},
            'expected_vuln_type': 'sql_injection',
            'cvss_range': [8.5, 9.3],  # FIX: was [7.5, 9.8] — much tighter range
            'expected_severity': 'critical',
            # FIX: More specific tokens — model must include ALL of these
            'required_fix_tokens': ['?', 'execute', 'param', 'parameterized'],
            'must_preserve_identifier': 'execute_query',
            'current_feedback_keywords': ['sql', 'injection', 'union'],
            'original_vuln_pattern': "query + user_input",
            'tool_call': 'db.execute("SELECT * FROM users WHERE name = \'" + user_input + "\'")',
            'task_description': 'Identify the SQL injection vulnerability in this MCP database tool call. Specify vuln_type, cvss_score (float), severity, and affected_line.',
        },
        {
            'case_id': 'sec_easy_002',
            'task_subtype': 'single',
            'completion_threshold': 0.75,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['identify_vulnerability']},
            'expected_vuln_type': 'xss',
            'cvss_range': [4.8, 5.8],  # FIX: very tight — model must know XSS CVSS precisely
            'expected_severity': 'medium',
            'required_fix_tokens': ['escape', 'html', 'sanitize'],
            'must_preserve_identifier': 'render_html',
            'current_feedback_keywords': ['xss', 'script', 'encoding'],
            'original_vuln_pattern': '<script>',
            'tool_call': 'render_html(f"<div>{user_comment}</div>")',
            'task_description': 'Scan this HTML rendering call for XSS vulnerability. Specify vuln_type, cvss_score (float, 0.0-10.0), severity, and affected_line.',
        },
        {
            'case_id': 'sec_easy_003',
            'task_subtype': 'single',
            'completion_threshold': 0.75,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['identify_vulnerability']},
            'expected_vuln_type': 'hardcoded_secret',
            'cvss_range': [6.2, 7.2],  # FIX: tighter range
            'expected_severity': 'high',
            'required_fix_tokens': ['env', 'os', 'getenv', 'environ'],
            'must_preserve_identifier': 'connect_db',
            'current_feedback_keywords': ['hardcoded', 'password', 'secret', 'credential'],
            'original_vuln_pattern': 'password="admin123"',
            'tool_call': 'connect_db(host="prod-db.internal", password="admin123", user="root")',
            'task_description': 'Scan this database connection call for hardcoded credentials. Specify vuln_type, cvss_score (float), severity, and affected_line.',
        },
        {
            'case_id': 'sec_easy_004',
            'task_subtype': 'single',
            'completion_threshold': 0.75,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['identify_vulnerability']},
            'expected_vuln_type': 'path_traversal',
            'cvss_range': [6.8, 7.8],  # FIX: tighter range
            'expected_severity': 'high',
            'required_fix_tokens': ['basename', 'path', 'normpath'],
            'must_preserve_identifier': 'read_file',
            'current_feedback_keywords': ['path', 'traversal', 'directory'],
            'original_vuln_pattern': '../../../etc/passwd',
            'tool_call': 'read_file(filepath=user_input)  # user sends "../../etc/passwd"',
            'task_description': 'Scan this file read call for path traversal vulnerability. Specify vuln_type, cvss_score (float), severity, and affected_line.',
        },
        {
            'case_id': 'sec_easy_005',
            'task_subtype': 'single',
            'completion_threshold': 0.75,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['identify_vulnerability']},
            'expected_vuln_type': 'missing_auth',
            'cvss_range': [7.8, 8.8],  # FIX: tighter range
            'expected_severity': 'critical',
            'required_fix_tokens': ['auth', 'token', 'verify', 'session'],
            'must_preserve_identifier': 'delete_user',
            'current_feedback_keywords': ['authentication', 'missing', 'unauthorized'],
            'original_vuln_pattern': 'no auth check',
            'tool_call': 'delete_user(user_id=request.params["id"])  # no authentication check',
            'task_description': 'Scan this admin action call for missing authentication. Specify vuln_type, cvss_score (float), severity, and affected_line.',
        },
    ],
    'sec_medium': [
        {
            'case_id': 'sec_medium_001',
            'task_subtype': 'multi',
            'completion_threshold': 0.65,  # FIX: was 0.75 — medium is harder to pass
            'max_steps': 6,
            # FIX: required_sequence now has ALL 3 actions — episode won't end until all done
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'sql_injection',
            'cvss_range': [8.8, 9.5],  # FIX: tighter range
            'expected_severity': 'critical',
            # FIX: More specific fix tokens — model must use parameterized queries specifically
            'required_fix_tokens': ['?', 'execute', 'param', 'parameterized', 'bind'],
            'must_preserve_identifier': 'search_products',
            'current_feedback_keywords': ['sql', 'union', 'injection', 'parameterize'],
            'original_vuln_pattern': 'query + ',
            'reviewer_feedback': 'Your fix prevents basic injection but UNION-based attacks are still possible. Use parameterized queries with bound parameters and add input length validation.',
            'additional_vulns': [
                {'vuln_type': 'xss', 'severity': 'medium', 'line': 15},
                {'vuln_type': 'missing_auth', 'severity': 'high', 'line': 3},
            ],
            'tool_call': '''def search_products(query, user_id):
    # No auth check on line 3
    results = db.execute("SELECT * FROM products WHERE name LIKE '%" + query + "%'")
    return f"<div>{results}</div>"  # XSS on line 15''',
            'task_description': 'This MCP tool has multiple security issues. Identify the most critical vulnerability, propose a fix, then revise based on reviewer feedback.',
        },
        {
            'case_id': 'sec_medium_002',
            'task_subtype': 'multi',
            'completion_threshold': 0.65,
            'max_steps': 6,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'ssrf',
            'cvss_range': [7.5, 8.5],  # FIX: tighter
            'expected_severity': 'high',
            'required_fix_tokens': ['allowlist', 'url', 'private', 'block'],
            'must_preserve_identifier': 'fetch_url',
            'current_feedback_keywords': ['ssrf', 'internal', 'url', 'private', 'ip'],
            'original_vuln_pattern': 'requests.get(user_url)',
            'reviewer_feedback': 'Fix addresses external URLs but internal network requests (169.254.x.x, localhost) are still allowed. Add an allowlist or explicitly block private IP ranges.',
            'additional_vulns': [
                {'vuln_type': 'hardcoded_secret', 'severity': 'medium', 'line': 2},
            ],
            'tool_call': '''def fetch_url(user_url):
    API_KEY = "sk-secret-key-12345"  # hardcoded secret
    resp = requests.get(user_url, headers={"Authorization": API_KEY})
    return resp.text''',
            'task_description': 'Scan this URL fetcher for vulnerabilities. Identify, propose a fix, then revise based on reviewer feedback.',
        },
        {
            'case_id': 'sec_medium_003',
            'task_subtype': 'multi',
            'completion_threshold': 0.65,
            'max_steps': 6,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'idor',
            'cvss_range': [6.5, 7.5],  # FIX: tighter
            'expected_severity': 'high',
            'required_fix_tokens': ['user_id', 'owner', 'session', 'authorization'],
            'must_preserve_identifier': 'get_document',
            'current_feedback_keywords': ['idor', 'authorization', 'object', 'session'],
            'original_vuln_pattern': 'doc_id from request',
            'reviewer_feedback': 'Fix checks user ownership but does not validate the requesting user session. Add session-based auth check and compare session user_id with document owner_id.',
            'additional_vulns': [
                {'vuln_type': 'rate_limit_missing', 'severity': 'medium', 'line': 1},
            ],
            'tool_call': '''@app.get("/documents/{doc_id}")  # no rate limiting
def get_document(doc_id: int, request: Request):
    # No check if requesting user owns this document
    return db.documents.find_one({"_id": doc_id})''',
            'task_description': 'Scan this document access endpoint for authorization issues. Identify, fix, then revise based on feedback.',
        },
    ],
    'sec_hard': [
        {
            'case_id': 'sec_hard_001',
            'task_subtype': 'adversarial',
            'completion_threshold': 0.60,  # FIX: was 0.70 — hard IS hard
            'max_steps': 8,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'sql_injection',
            'cvss_range': [9.0, 9.8],  # FIX: very tight — must score near-perfect on CVSS
            'expected_severity': 'critical',
            # FIX: More demanding fix tokens — model must use all of these
            'required_fix_tokens': ['?', 'execute', 'param', 'parameterized', 'validate', 'input'],
            'must_preserve_identifier': 'send_email',
            'current_feedback_keywords': ['still vulnerable', 'parameterize', 'concatenation', 'line 5'],
            'original_vuln_pattern': "query + user_input",
            'reviewer_feedback': 'Fix still allows SQL injection through the search parameter. Use parameterized queries for ALL string fields.',
            'reviewer_feedback_sequence': [
                'Fix still allows SQL injection through the search parameter. Use parameterized queries for ALL string fields.',
                'Better, but concatenation on line 5 still vulnerable. Also add input validation and length limits.',
            ],
            'tool_call': '''def send_email(recipient, subject, body):
    query = "INSERT INTO emails (to_addr, subject, body) VALUES ('" + recipient + "', '" + subject + "', '" + body + "')"
    db.execute(query)
    log(f"Email sent to {recipient}")
    return {"status": "sent", "to": recipient}''',
            'task_description': 'Rewrite this email tool call to remove SQL injection while preserving ALL business logic. A reviewer will evaluate your fix with adversarial inputs.',
        },
        {
            'case_id': 'sec_hard_002',
            'task_subtype': 'adversarial',
            'completion_threshold': 0.60,
            'max_steps': 8,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'jwt_misuse',
            'cvss_range': [8.5, 9.3],  # FIX: tighter
            'expected_severity': 'critical',
            # FIX: More demanding fix tokens
            'required_fix_tokens': ['verify', 'algorithm', 'HS256', 'secret', 'algorithms'],
            'must_preserve_identifier': 'authenticate_user',
            'current_feedback_keywords': ['jwt', 'algorithm', 'none', 'verify', 'signature'],
            'original_vuln_pattern': 'algorithms=["none"]',
            'reviewer_feedback': 'Fix does not prevent algorithm=none attack. Explicitly set allowed_algorithms=["HS256"] and use server-side secret.',
            'reviewer_feedback_sequence': [
                'Fix does not prevent algorithm=none attack. Explicitly set allowed_algorithms=["HS256"] and use server-side secret.',
                'Algorithm is set but secret key is still derived from user input. Use a hardcoded server secret from environment variables.',
            ],
            'tool_call': '''def authenticate_user(token):
    payload = jwt.decode(token, options={"verify_signature": False})
    user_id = payload.get("user_id")
    return get_user(user_id)''',
            'task_description': 'Rewrite this JWT authentication to prevent algorithm confusion attacks while preserving user lookup logic.',
        },
        {
            'case_id': 'sec_hard_003',
            'task_subtype': 'adversarial',
            'completion_threshold': 0.60,
            'max_steps': 8,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'xxe',
            'cvss_range': [7.8, 8.8],  # FIX: tighter
            'expected_severity': 'high',
            'required_fix_tokens': ['disable', 'external', 'entities', 'dtd', 'defusedxml'],
            'must_preserve_identifier': 'parse_xml',
            'current_feedback_keywords': ['xxe', 'entity', 'external', 'dtd', 'defused'],
            'original_vuln_pattern': 'allow_external_entities=True',
            'reviewer_feedback': "Fix disables DTD but doesn't disable external entities. Set both no_network=True and forbid_dtd=True, or use defusedxml.",
            'reviewer_feedback_sequence': [
                "Fix disables DTD but doesn't disable external entities. Set both no_network=True and forbid_dtd=True.",
                'Better, but you should also use defusedxml library for defense-in-depth and validate XML schema.',
            ],
            'tool_call': '''def parse_xml(xml_string):
    import xml.etree.ElementTree as ET
    parser = ET.XMLParser()  # Default parser allows XXE
    tree = ET.fromstring(xml_string, parser)
    return tree.findall('.//user/name')''',
            'task_description': 'Fix the XXE vulnerability in this XML parser. Reviewer will test with external entity payloads.',
        },
        {
            'case_id': 'sec_hard_004',
            'task_subtype': 'adversarial',
            'completion_threshold': 0.60,
            'max_steps': 8,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'ssrf',
            'cvss_range': [8.0, 9.0],  # FIX: tighter
            'expected_severity': 'critical',
            'required_fix_tokens': ['allowlist', 'private', 'ip', 'ipaddress', 'block'],
            'must_preserve_identifier': 'fetch_webhook',
            'current_feedback_keywords': ['ssrf', 'internal', 'network', 'private', 'ipv6'],
            'original_vuln_pattern': 'requests.get(url)',
            'reviewer_feedback': 'Fix blocks 127.0.0.1 but allows localhost and private IPs (10.x, 172.x, 192.168.x). Block ALL private ranges.',
            'reviewer_feedback_sequence': [
                'Fix blocks 127.0.0.1 but allows localhost and private IPs (10.x, 172.x, 192.168.x). Block ALL private ranges.',
                'Good, but you should also block IPv6 loopback (::1) and link-local addresses (fe80::).',
            ],
            'tool_call': '''def fetch_webhook(url):
    import requests
    # Webhook URL from user input
    response = requests.get(url, timeout=10)
    return response.json()''',
            'task_description': 'Prevent SSRF attacks on this webhook fetcher. Reviewer will try internal network access bypasses including IPv6.',
        },
        {
            'case_id': 'sec_hard_005',
            'task_subtype': 'adversarial',
            'completion_threshold': 0.60,
            'max_steps': 8,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'idor',
            'cvss_range': [7.0, 8.0],  # FIX: tighter
            'expected_severity': 'high',
            'required_fix_tokens': ['owner', 'session', 'user_id', 'token', 'verify'],
            'must_preserve_identifier': 'update_profile',
            'current_feedback_keywords': ['idor', 'authorization', 'owner', 'session', 'cryptographic'],
            'original_vuln_pattern': 'profile_id from request',
            'reviewer_feedback': 'Fix checks profile ownership but uses user_id from request body (attacker-controlled). Use session token, not request body user_id.',
            'reviewer_feedback_sequence': [
                'Fix checks profile ownership but uses user_id from request body (attacker-controlled). Use session token.',
                'Better, but session validation is weak. Use cryptographic session tokens, not just user_id in cookie.',
            ],
            'tool_call': '''@app.post("/profile/update")
def update_profile(profile_id: int, user_id: int, data: dict):
    # user_id comes from request body (!)
    profile = db.profiles.find_one({"_id": profile_id})
    profile.update(data)
    return {"status": "updated"}''',
            'task_description': 'Fix IDOR vulnerability allowing users to edit others\' profiles. Reviewer will test horizontal privilege escalation.',
        },
    ],
}
