## DatasetManager

[< back to parent (likelihoodInterfaceConfig)](./LikelihoodInterface.md)

Owns, configures and initializes the dataset definitions used to load model
and measurement events.

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| name | string | Optional identifier for configuration overrides | |
| datasetTemplateList | list | Named partial data dispenser configurations | Empty |
| datasetList | list | Dataset definitions, with their existing `name`, `model` and optional `data` fields | Empty |

### Dataset templates

`datasetTemplateList` holds named, partial data dispenser configurations. Template
fields are validated when the effective dispenser is configured.

```yaml
datasetManagerConfig:
  datasetTemplateList:
    - name: common
      tree: events
      filePathList: [input.root]
    - name: selection
      selectionCutFormula: "energy > 0"
  datasetList:
    - name: selectedEvents
      model:
        useTemplates:
          - name: common
          - name: selection
        tree: selected
      data:
        - name: measurement
          useTemplates:
            - name: common
          filePathList: [data.root]
      selectedDataEntry: measurement
```

Templates are applied in `useTemplates` order, followed by the dispenser's own
configuration, using the usual GUNDAM configuration override rules. A template's
`name` identifies the template and is not copied into the dispenser. References
are enabled by default; `isEnabled: false` skips the reference. Unknown enabled
template references and duplicate template names are errors. Templates do not
recursively use other templates.

`useTemplates` belongs to `model` or an entry of `data`, rather than to the
surrounding dataset definition. With `fromModel: true`, the effective model
configuration provides the base; the data templates and then the data entry's
own options override that base.
