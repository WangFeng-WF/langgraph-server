FROM langchain/langgraph-api:3.11-wolfi



ADD ./pip.conf /pipconfig.txt

# -- Adding local package . --
ADD . /deps/langgraph-server
# -- End of local package . --

# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt .);             fi;         done
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"agent": {"path": "/deps/langgraph-server/src/agent/graph.py:graph", "description": "Agent Graph"}, "procedure-agent": {"path": "/deps/langgraph-server/src/procedure-agent/graph.py:graph", "description": "Procedure Agent Graph"}, "manufacturing-assistant": {"path": "/deps/langgraph-server/src/manufacturing-assistant/manufacturing_assistant.py:manufacturing_assistant", "description": "Manufacturing Assistant Graph"}, "dbt-agent": {"path": "/deps/langgraph-server/src/dbt-core-agent/dbt_react_agent.py:analysis_agent_graph", "description": "DBT Agent Graph"}}'



# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx

WORKDIR /deps/langgraph-server