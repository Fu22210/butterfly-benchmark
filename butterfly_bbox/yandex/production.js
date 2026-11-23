function BoxProductionTask(options) {
  TolokaHandlebarsTask.call(this,options);
}
BoxProductionTask.prototype=Object.create(TolokaHandlebarsTask.prototype);
BoxProductionTask.prototype.validate=function (solution) {
  const out=(solution || this.getSolution()).output_values || {};
  const shapes=out.result || [];
  const fieldError=message=>({result:{code:"bbox_format",message}});
  if (!Array.isArray(shapes) || shapes.some(s=>s?.type!=="rectangle")) {
    return fieldError("Выделяйте объекты только прямоугольниками.");
  }
  if (out.status==="boxes" && !shapes.length) {
    return fieldError("Добавьте хотя бы одну рамку.");
  }
  if ((out.status==="none" || out.status==="load_error") && shapes.length) {
    return fieldError("Удалите рамки или измените статус ответа.");
  }
  return TolokaHandlebarsTask.prototype.validate.call(this,solution);
};
exports.Task=BoxProductionTask;
