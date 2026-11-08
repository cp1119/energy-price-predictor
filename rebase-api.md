# Use with an agent (https://grid.rebase.energy/docs/v1/guides/agents/)



An agent needs two things: one of the files below, and your API key
([Get a key](https://grid.rebase.energy/docs/v1/guides/authentication/#get-a-key) says how to get one).

Three machine-readable entry points are published with these docs:

| File                                                         | Contents                                      |
| ------------------------------------------------------------ | --------------------------------------------- |
| [`grid.rebase.energy/docs/v1/openapi.json`](https://grid.rebase.energy/docs/v1/openapi.json)   | the endpoint contract, with example responses |
| [`grid.rebase.energy/docs/v1/llms.txt`](https://grid.rebase.energy/docs/v1/llms.txt)           | an index of every page of these docs          |
| [`grid.rebase.energy/docs/v1/llms-full.txt`](https://grid.rebase.energy/docs/v1/llms-full.txt) | every page of these docs as one Markdown file |

Every page is also available as Markdown from the **Copy Markdown** button at its top.

The entity tree gives an agent the ids and how they nest, each response carries the path of the
next call (`metadata`, `values`, `list`), and once the entities are known search finds a series in
one request, so an agent with the key and one of these files can collect a dataset on its own. [Calling the API efficiently](https://grid.rebase.energy/docs/v1/guides/efficient-calls/) lists the rules for doing it in few
calls and without errors; `llms.txt` starts with them. Example prompt:

