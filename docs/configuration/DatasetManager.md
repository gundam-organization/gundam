## DatasetManager

[< back to parent (likelihoodInterfaceConfig)](./LikelihoodInterface.md)

Owns, configures and initializes the dataset definitions used to load model
and measurement events.

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| name | string | Optional identifier for configuration overrides | |
| datasetTemplateList | list | Named partial dataset configurations | Empty |
| datasetList | list | Dataset definitions, with their existing `name`, `model` and optional `data` fields | Empty |

### Dataset templates

`datasetTemplateList` holds named, partial dataset configurations. Template
fields are validated when the effective dataset is configured.

```yaml
datasetManagerConfig:
  datasetTemplateList:
    - name: common
      model:
        tree: events
        filePathList: [input.root]
    - name: selection
      model:
        selectionCutFormula: "energy > 0"
  datasetList:
    - name: selectedEvents
      useTemplates:
        - name: common
        - name: selection
          isEnabled: true
      model:
        tree: selected
```

Templates are applied in `useTemplates` order, followed by the dataset's own
configuration, using the usual GUNDAM configuration override rules. A template's
`name` identifies the template and is not copied into the dataset. References
are enabled by default; `isEnabled: false` skips the reference. Unknown enabled
template references and duplicate template names are errors. Templates do not
recursively use other templates.
