package com.zzk;

import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Element;
import com.itextpdf.text.Phrase;
import com.itextpdf.text.pdf.PdfPCell;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class AbsoluteOrientationTable {
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("c:\\绝对定位表格.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            float[] columnSize = { 21F, 21F, 21F };// 设置列宽
            PdfPTable table = null;
            PdfPCell cell = null;
            table = new PdfPTable(columnSize);// 定义新表格
            table.getDefaultCell().setBorder(1);// 设置表格边框宽度
            table.setHorizontalAlignment(Element.ALIGN_CENTER);// 设置居中对齐
            table.setTotalWidth(500); // 设置总宽度500
            table.setLockedWidth(true); // 解锁
            cell = new PdfPCell(new Phrase("Add table"));// 定义单元格
            cell.setColspan(3);// 设置单元格跨度3
            table.addCell(cell);// 向表格添加单元格
            table.addCell(new PdfPCell(new Phrase("Add 001")));// 向表格添加内容
            table.addCell(new PdfPCell(new Phrase("Add 002")));// 向表格添加内容
            table.addCell(new PdfPCell(new Phrase("Add 003")));// 向表格添加内容
            document.add(table);// 向文档添加表格
            table = new PdfPTable(columnSize);
            // 定义新表格
            table.getDefaultCell().setBorder(1);// 设置表格边框宽度
            table.setHorizontalAlignment(Element.ALIGN_CENTER);// 设置居中对齐
            table.setTotalWidth(500);// 设置总宽度500
            table.setLockedWidth(true);// 解锁
            cell = new PdfPCell(new Phrase("Table writeSelectedRows"));// 定义单元格
            cell.setColspan(columnSize.length);// 设置单元格跨度3
            table.addCell(cell); // 向表格添加单元格
            table.addCell(new PdfPCell(new Phrase("Add 004")));// 向表格添加内容
            table.addCell(new PdfPCell(new Phrase("Add 005")));// 向表格添加内容
            table.addCell(new PdfPCell(new Phrase("Add 006")));// 向表格添加内容
            table.writeSelectedRows(0, 2, 50, 750, writer.getDirectContent());// 在指定位置添加表格内容
            document.close();// 关闭文档
        } catch (DocumentException de) {
        } catch (IOException ioe) {
            
        }
        
    }
}
